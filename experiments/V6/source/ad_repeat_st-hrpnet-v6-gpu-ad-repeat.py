import os, json, hashlib, math, gc, time
from pathlib import Path
import numpy as np, pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import average_precision_score
SEED=42
OUT=Path('/kaggle/working/ST_HRPNet_V6_GPU_AD_repeat')
OUT.mkdir(exist_ok=True)
root=next(p.parent for p in Path('/kaggle/input').rglob('sequences_lossless.npz'))
packed=np.load(root/'sequences_lossless.npz')
frames=packed['frames']
packing=json.loads((root/'packing_audit.json').read_text())
clean={}; seen=set(); hashes=set(); audit={}
for s in ('train','val','test'):
    x=frames[packed[f'index_{s}']]; y=packed[f'y_{s}']
    assert hashlib.sha256(x.tobytes()).hexdigest()==packing['splits'][s]['x_sha256']
    assert hashlib.sha256(y.tobytes()).hexdigest()==packing['splits'][s]['y_sha256']
    m=pd.read_csv(root/f'meta_{s}.csv',dtype=str)
    assert x.shape==(len(m),4,256,256,4) and y.shape==(len(m),256,256,1)
    keep=[]; current=set(); current_hash=set()
    for i,r in m.iterrows():
        codes=r.sequence_time_codes.split(',')
        assert len(codes)==4 and codes[-1]==r.target_time_code
        mins=[int(c[:2])*60+int(c[2:]) for c in codes]
        if any(b-a!=10 for a,b in zip(mins,mins[1:])) or seen.intersection(codes): continue
        h={hashlib.sha256(np.ascontiguousarray(f).tobytes()).hexdigest() for f in x[i]}
        if h.intersection(hashes): continue
        assert np.isfinite(x[i]).all() and np.isin(y[i],[0,1]).all()
        keep.append(i); current.update(codes); current_hash.update(h)
    seen.update(current); hashes.update(current_hash)
    clean[s]=(np.array(x[keep],dtype=np.float32),np.array(y[keep],dtype=np.float32))
    audit[s]={'original':len(m),'retained':len(keep),'indices':keep,'positive_ratio':float(clean[s][1].mean())}
    assert 0<clean[s][1].mean()<1
    m.iloc[keep].to_csv(OUT/f'meta_{s}.csv',index=False)
(OUT/'split_audit.json').write_text(json.dumps(audit,indent=2))
print(audit,flush=True)
assert [audit[s]['retained'] for s in ('train','val','test')]==[91,18,18], 'Unexpected split; stop for audit'
packed.close(); del frames,x,y; gc.collect()
gpus=tf.config.list_physical_devices('GPU')
assert gpus, 'GPU unavailable: no CPU/TPU fallback'
for g in gpus: tf.config.experimental.set_memory_growth(g,True)
strategy=tf.distribute.MirroredStrategy() if len(gpus)>1 else tf.distribute.OneDeviceStrategy('/GPU:0')
keras.mixed_precision.set_global_policy('mixed_float16')
BATCH=8
assert BATCH%strategy.num_replicas_in_sync==0
assert 1800%BATCH==0 and 448%BATCH==0
SEQ_LEN=4; INPUT_PRODUCTS=['b13','b08','b07','b03']; BASE_FILTERS=16; BATCH_SIZE=BATCH
LOSS_POS_WEIGHT=float(np.clip((1-clean['train'][1].mean())/(clean['train'][1].mean()+1e-7),1,20))
print('GPU devices',gpus,'replicas',strategy.num_replicas_in_sync,'batch',BATCH,flush=True)
# ============================================================
# 7. LOSS FUNCTIONS, DATASET BUILDER, EXACT ST-HRPNET BUILDER
# ============================================================

SMOOTH = 1e-6

if "LOSS_POS_WEIGHT" not in globals():
    source_y = globals().get("y_train_patch", globals().get("y_train", None))
    if source_y is not None:
        pos_ratio = float(np.mean(source_y))
        neg_ratio = 1.0 - pos_ratio
        LOSS_POS_WEIGHT = min(max(neg_ratio / (pos_ratio + 1e-7), 1.0), 20.0)
    else:
        LOSS_POS_WEIGHT = 6.815723822580276

print("LOSS_POS_WEIGHT:", LOSS_POS_WEIGHT)


def dice_coef(y_true, y_pred):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    inter = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * inter + SMOOTH) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + SMOOTH
    )


def iou_coef(y_true, y_pred):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    inter = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - inter
    return (inter + SMOOTH) / (union + SMOOTH)


def weighted_bce_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-6, 1.0 - 1e-6)
    bce = keras.backend.binary_crossentropy(y_true, y_pred)
    weights = y_true * LOSS_POS_WEIGHT + (1.0 - y_true)
    return tf.reduce_mean(bce * weights)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def false_positive_penalty(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    bg = 1.0 - y_true
    return tf.reduce_sum(bg * y_pred) / (tf.reduce_sum(bg) + SMOOTH)


def combined_hrp_loss(y_true, y_pred):
    return (
        0.60 * weighted_bce_loss(y_true, y_pred)
        + 0.35 * dice_loss(y_true, y_pred)
        + 0.05 * false_positive_penalty(y_true, y_pred)
    )


def combined_hrp_loss_no_fp(y_true, y_pred):
    return 0.65 * weighted_bce_loss(y_true, y_pred) + 0.35 * dice_loss(y_true, y_pred)


def precision_at_05(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred_bin = tf.cast(y_pred >= 0.5, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred_bin)
    fp = tf.reduce_sum((1.0 - y_true) * y_pred_bin)
    return (tp + SMOOTH) / (tp + fp + SMOOTH)


def recall_at_05(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred_bin = tf.cast(y_pred >= 0.5, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred_bin)
    fn = tf.reduce_sum(y_true * (1.0 - y_pred_bin))
    return (tp + SMOOTH) / (tp + fn + SMOOTH)


def compile_model_with_loss(model, loss_fn="with_fp", lr=7e-4):
    if loss_fn == "no_fp":
        loss = combined_hrp_loss_no_fp
    elif loss_fn == "bce":
        loss = "binary_crossentropy"
    else:
        loss = combined_hrp_loss

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=[dice_coef, iou_coef, precision_at_05, recall_at_05],
    )
    return model


def make_dataset(X_arr, y_arr, batch_size=BATCH_SIZE, shuffle=False):
    X_arr = np.asarray(X_arr, dtype=np.float32)
    y_arr = np.asarray(y_arr, dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((X_arr, y_arr))

    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(len(X_arr), 2048),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ============================================================
# EXACT ARCHITECTURE FROM MAIN TRAINING NOTEBOOK
# ============================================================

def td_conv_block(x, filters, name, dropout=0.0):
    x = layers.TimeDistributed(
        layers.SeparableConv2D(
            filters,
            3,
            padding="same",
            use_bias=False,
        ),
        name=f"{name}_sepconv1",
    )(x)

    x = layers.TimeDistributed(
        layers.BatchNormalization(),
        name=f"{name}_bn1",
    )(x)

    x = layers.TimeDistributed(
        layers.Activation("relu"),
        name=f"{name}_relu1",
    )(x)

    x = layers.TimeDistributed(
        layers.SeparableConv2D(
            filters,
            3,
            padding="same",
            use_bias=False,
        ),
        name=f"{name}_sepconv2",
    )(x)

    x = layers.TimeDistributed(
        layers.BatchNormalization(),
        name=f"{name}_bn2",
    )(x)

    x = layers.TimeDistributed(
        layers.Activation("relu"),
        name=f"{name}_relu2",
    )(x)

    if dropout > 0:
        x = layers.TimeDistributed(
            layers.SpatialDropout2D(dropout),
            name=f"{name}_drop",
        )(x)

    return x


def conv_block_2d(x, filters, name, dropout=0.0):
    x = layers.SeparableConv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        name=f"{name}_sepconv1",
    )(x)

    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)

    x = layers.SeparableConv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        name=f"{name}_sepconv2",
    )(x)

    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_relu2")(x)

    if dropout > 0:
        x = layers.SpatialDropout2D(dropout, name=f"{name}_drop")(x)

    return x


def last_time_step(x):
    return x[:, -1, ...]


def build_st_hrpnet_v1_patchbalanced(
    seq_len=SEQ_LEN,
    n_channels=len(INPUT_PRODUCTS),
    base=BASE_FILTERS,
):
    inputs = keras.Input(
        shape=(seq_len, SIZE, SIZE, n_channels),
        name="himawari_sequence_dynamic",
    )

    x1 = td_conv_block(inputs, base, "enc1", dropout=0.03)
    p1 = layers.TimeDistributed(
        layers.MaxPooling2D((2, 2)),
        name="pool1",
    )(x1)

    x2 = td_conv_block(p1, base * 2, "enc2", dropout=0.05)
    p2 = layers.TimeDistributed(
        layers.MaxPooling2D((2, 2)),
        name="pool2",
    )(x2)

    x3 = td_conv_block(p2, base * 4, "enc3", dropout=0.08)
    p3 = layers.TimeDistributed(
        layers.MaxPooling2D((2, 2)),
        name="pool3",
    )(x3)

    b = layers.ConvLSTM2D(
        base * 8,
        3,
        padding="same",
        return_sequences=False,
        dropout=0.15,
        recurrent_dropout=0.0,
        name="conv_lstm_bottleneck", dtype="float32",
    )(p3)

    b = layers.BatchNormalization(name="bottleneck_bn")(b)
    b = layers.Activation("relu", name="bottleneck_relu")(b)

    s3 = layers.Lambda(last_time_step, name="skip3_last")(x3)
    s2 = layers.Lambda(last_time_step, name="skip2_last")(x2)
    s1 = layers.Lambda(last_time_step, name="skip1_last")(x1)

    u3 = layers.UpSampling2D((2, 2), name="up3")(b)
    u3 = layers.Concatenate(name="cat3")([u3, s3])
    d3 = conv_block_2d(u3, base * 4, "dec3", dropout=0.05)

    u2 = layers.UpSampling2D((2, 2), name="up2")(d3)
    u2 = layers.Concatenate(name="cat2")([u2, s2])
    d2 = conv_block_2d(u2, base * 2, "dec2", dropout=0.03)

    u1 = layers.UpSampling2D((2, 2), name="up1")(d2)
    u1 = layers.Concatenate(name="cat1")([u1, s1])
    d1 = conv_block_2d(u1, base, "dec1", dropout=0.02)

    outputs = layers.Conv2D(
        1,
        1,
        activation="sigmoid",
        name="hrp_mask", dtype="float32",
    )(d1)

    return keras.Model(
        inputs,
        outputs,
        name="ST_HRPNet_v1_PatchBalanced",
    )


print("Exact ST-HRPNet v1 architecture loaded.")
print("BASE_FILTERS:", BASE_FILTERS)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def make_patches(split,n,frac,seed):
    x,y=clean[split]; rng=np.random.default_rng(seed)
    eligible=[i for i in range(len(y)) if y[i].sum()>0]
    assert eligible
    a=np.empty((n,4,128,128,4),np.float32); b=np.empty((n,128,128,1),np.float32); records=[]
    for j in range(n):
        positive=j>=n-int(n*frac)
        if positive:
            i=int(rng.choice(eligible)); pts=np.argwhere(y[i,:,:,0]>0); cy,cx=pts[rng.integers(len(pts))]
            t=int(np.clip(cy-64,0,128)); l=int(np.clip(cx-64,0,128))
        else:
            i=int(rng.integers(len(y))); t=int(rng.integers(129)); l=int(rng.integers(129))
        a[j]=x[i,:,t:t+128,l:l+128]; b[j]=y[i,t:t+128,l:l+128]
        if positive: assert b[j].sum()>0
        records.append((i,t,l,positive,float(b[j].mean())))
    return a,b,pd.DataFrame(records,columns=['sequence','top','left','positive_centered','positive_ratio'])
def ds(x,y,shuffle=False):
    with tf.device('/CPU:0'):
        d=tf.data.Dataset.from_tensor_slices((x,y))
        if shuffle: d=d.shuffle(min(len(x),128),seed=SEED)
        return d.batch(BATCH,drop_remainder=True).prefetch(2)
def metric(y,p,t):
    y=y.ravel()>0.5; z=p.ravel()>=t
    tp=int((y&z).sum()); fp=int((~y&z).sum()); fn=int((y&~z).sum()); tn=int((~y&~z).sum())
    div=lambda a,b:a/b if b else 0.
    rec=div(tp,tp+fn); spec=div(tn,tn+fp)
    return dict(threshold=float(t),iou=div(tp,tp+fp+fn),dice_f1=div(2*tp,2*tp+fp+fn),precision=div(tp,tp+fp),recall=rec,specificity=spec,fpr=1-spec,balanced_accuracy=(rec+spec)/2,mcc=div(tp*tn-fp*fn,math.sqrt(float(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))),tp=tp,fp=fp,fn=fn,tn=tn,actual_ratio=float(y.mean()),pred_ratio=float(z.mean()))
def choose(y,p):
    rows=[]
    for t in np.round(np.arange(.05,.951,.01),2):
        m=metric(y,p,t); r=m['actual_ratio']; q=m['pred_ratio']
        m.update(valid=.5*r<=q<=2*r,score=m['iou']+.2*m['dice_f1']+.1*m['balanced_accuracy']+.1*m['mcc']-.025*abs(np.log((q+1e-7)/(r+1e-7))))
        rows.append(m)
    d=pd.DataFrame(rows); v=d[d.valid]; return float((v if len(v) else d).sort_values(['score','threshold'],ascending=[False,True]).iloc[0].threshold),d
def infer(m,x):
    n=len(x); pad=(-n)%BATCH
    if pad: x=np.concatenate([x,np.repeat(x[-1:],pad,axis=0)])
    p=m.predict(tf.data.Dataset.from_tensor_slices(x).batch(BATCH,drop_remainder=True),verbose=0)[:n]
    assert np.isfinite(p).all() and p.shape==(n,256,256,1)
    return p
def build(size):
    global SIZE
    SIZE=size
    with strategy.scope(): return build_st_hrpnet_v1_patchbalanced()
def loss_for(v):
    def loss(y,p):
        y=tf.cast(y,tf.float32); p=tf.clip_by_value(tf.cast(p,tf.float32),1e-6,1-1e-6)
        ax=(1,2,3)
        bce=tf.reduce_mean(-(LOSS_POS_WEIGHT*y*tf.math.log(p)+(1-y)*tf.math.log1p(-p)),axis=ax)
        dice=1-(2*tf.reduce_sum(y*p,axis=ax)+1e-6)/(tf.reduce_sum(y+p,axis=ax)+1e-6)
        fp=tf.reduce_sum((1-y)*p,axis=ax)/(tf.reduce_sum(1-y,axis=ax)+1e-6)
        return .65*bce+.35*dice if v=='A' else .60*bce+.35*dice+.05*fp
    return loss
class ForegroundIoU(keras.metrics.Metric):
    def __init__(self,name='hrp_iou',**kwargs):
        super().__init__(name=name,**kwargs); self.inter=self.add_weight(name='inter',initializer='zeros'); self.union=self.add_weight(name='union',initializer='zeros')
    def update_state(self,y,p,sample_weight=None):
        y=tf.cast(y>=.5,tf.float32); p=tf.cast(p>=.5,tf.float32)
        self.inter.assign_add(tf.reduce_sum(y*p)); self.union.assign_add(tf.reduce_sum(y+p-y*p))
    def result(self): return tf.math.divide_no_nan(self.inter,self.union)
    def reset_state(self): self.inter.assign(0.); self.union.assign(0.)
def compiled(v):
    m=build(128)
    with strategy.scope(): m.compile(optimizer=keras.optimizers.Adam(1e-4),loss=loss_for(v),metrics=[ForegroundIoU()],jit_compile=False)
    return m
keras.utils.set_random_seed(SEED)
sm=compiled('D')
sx=np.zeros((BATCH,4,128,128,4),np.float32); sy=np.zeros((BATCH,128,128,1),np.float32); sy[:,40:48,40:48]=1
res=sm.fit(ds(sx,sy),epochs=1,verbose=2)
assert np.isfinite(res.history['loss']).all()
(OUT/'GPU_SMOKE_OK.json').write_text(json.dumps({'replicas':strategy.num_replicas_in_sync,'batch':BATCH,'gpus':[str(g) for g in gpus],'tf':tf.__version__,'loss':res.history['loss']},indent=2))
del sm; keras.backend.clear_session(); gc.collect()
vx,vy,vm=make_patches('val',448,.5,244); vm.to_csv(OUT/'common_validation_patches.csv',index=False)
vd=ds(vx,vy)
rows=[]; probs={}; thresholds={}
for seed,v in [(123,'A'),(123,'D'),(2025,'A'),(2025,'D')]:
    SEED=seed
    keras.backend.clear_session(); gc.collect(); keras.utils.set_random_seed(SEED)
    r=OUT/f'{v}_seed{SEED}'; r.mkdir(exist_ok=True)
    x,y,pm=make_patches('train',1800,0 if v=='A' else .5,SEED+101); pm.to_csv(r/'patches.csv',index=False)
    m=compiled(v)
    with open(r/'model_summary.txt','w') as f: m.summary(print_fn=lambda s,**kwargs:f.write(s+'\n'))
    callbacks=[keras.callbacks.TerminateOnNaN(),keras.callbacks.ModelCheckpoint(str(r/'best.weights.h5'),monitor='val_hrp_iou',mode='max',save_best_only=True,save_weights_only=True),keras.callbacks.EarlyStopping(monitor='val_hrp_iou',mode='max',patience=10),keras.callbacks.ReduceLROnPlateau(monitor='val_hrp_iou',mode='max',patience=5,factor=.5,min_lr=1e-6),keras.callbacks.CSVLogger(str(r/'history.csv'))]
    start=time.time(); h=m.fit(ds(x,y,True),validation_data=vd,epochs=40,callbacks=callbacks,verbose=2)
    assert all(np.isfinite(a).all() for a in h.history.values())
    assert (r/'best.weights.h5').exists()
    (r/'TRAINING_COMPLETE.json').write_text(json.dumps({'epochs':len(h.history['loss']),'elapsed_seconds':time.time()-start,'parameters':m.count_params()}))
    del m,x,y; gc.collect()
    m=build(256); m.load_weights(str(r/'best.weights.h5'))
    p=infer(m,clean['val'][0]); np.save(r/'validation_prob.npy',p)
    t,d=choose(clean['val'][1],p); d.to_csv(r/'threshold_validation.csv',index=False); thresholds[f'{v}_seed{SEED}']=t
    p=infer(m,clean['test'][0]); np.save(r/'test_prob.npy',p); probs[f'{v}_seed{SEED}']=p
    yt=clean['test'][1]; ap=float(average_precision_score(yt.ravel(),p.ravel()))
    scenes=[]
    for setting,th in [('fixed',.5),('validation_selected',t)]:
        z=metric(yt,p,th); z.update(variant=v,seed=SEED,setting=setting,average_precision=ap); rows.append(z)
        for i in range(len(yt)):
            z=metric(yt[i],p[i],th); z.update(scene=i,seed=SEED,setting=setting); scenes.append(z)
    pd.DataFrame(scenes).to_csv(r/'per_scene.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'pilot_metrics.csv',index=False)
    del m; gc.collect()
np.save(OUT/'test_masks.npy',clean['test'][1])
fig,axs=plt.subplots(3,4,figsize=(10,7))
for a,i in zip(axs,[0,8,17]):
    for ax,img,title in zip(a,[clean['test'][0][i,-1,:,:,0],clean['test'][1][i,:,:,0],probs['A_seed123'][i,:,:,0]>=thresholds['A_seed123'],probs['D_seed123'][i,:,:,0]>=thresholds['D_seed123']],['B13','HRP','A','D']):
        ax.imshow(img,cmap='gray'); ax.set_title(title); ax.axis('off')
fig.tight_layout(); fig.savefig(OUT/'comparison.png',dpi=300)
(OUT/'PILOT_COMPLETE.json').write_text(json.dumps({'thresholds':thresholds,'split':audit,'seeds':[123,2025],'replicas':strategy.num_replicas_in_sync,'batch':BATCH,'precision':'mixed_float16; float32 recurrence and output','protocol':'V6 exploratory; shared validation patches; per-example loss; no event-independent claim'}))
print(pd.DataFrame(rows).to_string(index=False),flush=True)
