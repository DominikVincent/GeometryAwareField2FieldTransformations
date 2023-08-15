set -x          # activate debugging from here

ns-train nerfacto \
	--load-config $1 \
	--output-dir /tmp \
	--viewer.start-train False \
	--viewer.websocket-port 7009 \
	--vis viewer
