const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const path = require('path');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

app.use(express.static('public'));

const MAX_MESSAGE_BYTES = 5 * 1024 * 1024;

wss.on('connection', (ws) => {
  console.log('New client connected');
  ws.isAlive = true;

  ws.on('pong', () => {
    ws.isAlive = true;
  });

  ws.on('message', (message, isBinary) => {
    try {
      if (isBinary) return;

      const data = JSON.parse(message);
      broadcastMessage(data);

    } catch (error) {
      console.error('파싱 오류:', error);
    }
  });

  ws.on('close', () => {
    console.log('클라이언트 연결 종료');
  });
});

function broadcastMessage(message) {
  const msgString = JSON.stringify(message);
  wss.clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(msgString);
    }
  });
}

const interval = setInterval(() => {
  wss.clients.forEach((ws) => {
    if (!ws.isAlive) {
      console.log('비활성 클라이언트 종료');
      return ws.terminate();
    }
    ws.isAlive = false;
    ws.ping(() => {});
  });
}, 30000);

process.on('SIGTERM', () => {
  clearInterval(interval);
  server.close(() => {
    process.exit(0);
  });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`서버 실행 중: http://localhost:${PORT}`);
});

