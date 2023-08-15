set -x          # activate debugging from here

ns-train nerfacto \
	--trainer.load-dir $2 \
	--trainer.load-step $3 \
	--output-dir /tmp \
	--viewer.start-train False \
	--trainer.save-only-latest-checkpoint False \
	--pipeline.datamanager.camera-optimizer.mode off \
	--data $1 \
	nerfstudio-data
