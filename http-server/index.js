const API_KEY = process.env.API_KEY;

if (!API_KEY) {
  console.error("API_KEY is not set");
  process.exit(1);
}

const admin = require("firebase-admin");

const serviceAccount = require("./sa.json");

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
  databaseURL: "https://mottag-acbda-default-rtdb.firebaseio.com/",
});

const db = admin.database();

const express = require("express");
const rateLimit = require("express-rate-limit");

const app = express();

const limiter = rateLimit({
  windowMs: 1 * 60 * 1000, // 1 minute
  max: 100, // limit each IP to 100 requests per windowMs
  message: { error: "Too many requests, please try again later." },
});

app.use(limiter);

app.use((req, res, next) => {
  const key = req.headers["x-api-key"];
  if (key !== API_KEY) {
    return res.status(401).json({ error: "Unauthorized" });
  }
  next();
});

app.use(express.json());

app.post("/feed", (req, res) => {
  const antenaId = req.body.aid;
  console.log(`Received payload for antenaId ${antenaId}:`, req.body);

  const data = {
    hardware_id: req.body.addr.toUpperCase(),
    rssi: req.body.rssi,
    board_timestamp: req.body.time,
    server_timestamp: admin.database.ServerValue.TIMESTAMP,
  };

  const dispositivosRef = db.ref(`feed/${data.hardware_id}/${antenaId}`);
  dispositivosRef.set(data);

  res.sendStatus(200);
});

app.post("/feed2", (req, res) => {
  const antenaId = req.body.aid;
  console.log(`Received payload for antenaId ${antenaId}:`, req.body);

  const events = req.body.events;
  const promises = [];

  for (const event of events) {
    const hardwareId = event.addr.toUpperCase();
    const eventData = {
      server_timestamp: admin.database.ServerValue.TIMESTAMP,
      rssi: event.rssi,
      board_timestamp: event.t,
    };

    const dispositivosRef = db.ref(`feed/${hardwareId}/${antenaId}`);
    promises.push(dispositivosRef.set(eventData));
  }

  Promise.all(promises)
    .then(() => {
      res.sendStatus(200);
    })
    .catch((error) => {
      console.error("Error saving data:", error);
      res.status(500).json({ error: "Internal Server Error" });
    });
});

const PORT = 80;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
