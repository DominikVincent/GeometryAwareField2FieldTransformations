set -x          # activate debugging from here

ns-train nerfacto \
	--load-dir $2 \
	--load-step $3 \
	--output-dir /tmp \
	--viewer.start-train False \
	--save-only-latest-checkpoint False \
	--data $1 \
	--viewer.websocket-port 7015
