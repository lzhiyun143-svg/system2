Getting started
1 Start the backend
python -m uvicorn web_server:app --host 127.0.0.1 --port 8000 --reload
2 launch the front end
cd rehab-web
npm run dev
3 The server launches the musetalk model
uvicorn musetalk_api:app --host 0.0.0.0 --port 52595
4 Establish connection between server and the local machine
ssh -L 19000:127.0.0.1:52595 server11
$env:MUSETALK_API_BASE="http://127.0.0.1:19000"
3 Open the web
http://localhost:5173/


ssh -L 19000:127.0.0.1:19002 server11

uvicorn mimicmotion_api:app --host 0.0.0.0 --port 19002