set HTTP_PROXY=http://192.168.31.98:1080
set HTTPS_PROXY=http://192.168.31.98:1080
cd ../
docker build ./ -f ./Qwen2_5-VL/Dockerfile --build-arg HTTP_PROXY=http://192.168.31.98:1080 --build-arg HTTPS_PROXY=http://192.168.31.98:1080 -t lianshufeng/model_api:qwen2_5vl
pause