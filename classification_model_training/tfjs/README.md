
```bash
npm install @tensorflow/tfjs-node
npm install @tensorflow/tfjs-node-gpu
```

```bash
./eval.js \
    PATH_TO_DATASET/test \
    PATH_TO_MODEL/model.json \
    --gpu \
    --normalize none
```

```bash
./eval.js \
    PATH_TO_DATASET/test \
    PATH_TO_MODEL/model.json \
    --gpu \
    --normalize MeanStd \
    --dataFormat channels_first \
    --mean 0.485,0.456,0.406 \
    --std 0.229,0.224,0.225 \
    --div 255.0
```

```bash
./eval.js \
    PATH_TO_DATASET/test \
    PATH_TO_MODEL/model.json \
    --gpu \
    --normalize MeanStd \
    --mean 103.939,116.779,123.68
```
