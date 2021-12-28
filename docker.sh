sudo docker build -t mrcnn_server .
sudo docker rm -f server
sudo docker run --network host --name server mrcnn_server