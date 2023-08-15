

/data/vision/polina/projects/wmh/dhollidt/conda/envs/nerfstudio2/bin/ns-train depth-nerfacto \
    --vis viewer \
    --pipeline.datamanager.camera-optimizer.mode off \
    --save-only-latest-checkpoint True \
    --pipeline.model.predict-normals True \
    --max-num-iterations 8000 --data /data/vision/polina/projects/wmh/dhollidt/datasets/toybox-5/0 \
    --experiment-name 0 \
    --output-dir /data/vision/polina/projects/wmh/dhollidt/documents/nerf/nesf_models/ \
    --pipeline.model.pred-normal-loss-mult 0.01 \
    --pipeline.model.is-euclidean-depth True \
    --pipeline.model.max-res 1024 \
    --viewer.skip-openrelay True \
    --timestamp 07_04_23
