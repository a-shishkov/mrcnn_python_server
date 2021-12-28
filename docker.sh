docker build -t mrcnn_server .
docker rm -f server
docker run --network host --name server mrcnn_server