// TODO: Replace with your Firebase config
const firebaseConfig = {
  apiKey: "AIzaSyAHY_c4edguRyxtI14Iqo2wlFwu3e7pj9g",
  authDomain: "mottag-acbda.firebaseapp.com",
  databaseURL: "https://mottag-acbda-default-rtdb.firebaseio.com",
  projectId: "mottag-acbda",
  storageBucket: "mottag-acbda.firebasestorage.app",
  messagingSenderId: "936689121972",
  appId: "1:936689121972:web:d5421eb9d11dc811b975e1"
};

firebase.initializeApp(firebaseConfig);
const db = firebase.database();

const canvas = document.getElementById('mapCanvas');
const ctx = canvas.getContext('2d');

let antenasData = {};

// Listen for fixed points (antenas)
db.ref('antenas').on('value', snapshot => {
    antenasData = snapshot.val() || {};
});

// Listen for real-time updates
db.ref('posicoes').on('value', snapshot => {
    const positions = snapshot.val() || {};
    drawMap(positions);
});

function drawMap(positions) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Draw grid (optional)
    ctx.strokeStyle = '#eee';
    for (let x = 0; x < canvas.width; x += 50) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
    }
    for (let y = 0; y < canvas.height; y += 50) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
    }
    // Offset for origin
    const OFFSET_X = 100; // pixels to the right
    const OFFSET_Y = 100; // pixels up
    // Draw X axis (bottom, with offset)
    ctx.beginPath();
    ctx.moveTo(OFFSET_X, canvas.height - OFFSET_Y);
    ctx.lineTo(canvas.width, canvas.height - OFFSET_Y);
    ctx.strokeStyle = '#ff0000';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.font = '16px Arial';
    ctx.fillStyle = '#ff0000';
    ctx.fillText('X', canvas.width - 20, canvas.height - OFFSET_Y - 10);

    // Draw Y axis (left, with offset)
    ctx.beginPath();
    ctx.moveTo(OFFSET_X, 0);
    ctx.lineTo(OFFSET_X, canvas.height);
    ctx.strokeStyle = '#00aa00';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.font = '16px Arial';
    ctx.fillStyle = '#00aa00';
    ctx.fillText('Y', OFFSET_X + 10, 20);
    ctx.lineWidth = 1; // Reset line width
    // Draw positions
    const MULTIPLIER = 40; // Change this value to adjust scaling

    // Draw fixed points from antenasData
    Object.values(antenasData).forEach(pt => {
        const fx = OFFSET_X + pt.x * MULTIPLIER;
        const fy = canvas.height - OFFSET_Y - (pt.y * MULTIPLIER);
        ctx.beginPath();
        ctx.arc(fx, fy, 8, 0, 2 * Math.PI);
        ctx.fillStyle = '#ff9900';
        ctx.fill();
        ctx.strokeStyle = '#333';
        ctx.stroke();
        ctx.font = '12px Arial';
        ctx.fillStyle = '#ff9900';
        ctx.fillText(pt.id, fx + 10, fy - 10);
    });

    // Draw dynamic positions
    Object.values(positions).forEach(pos => {
        // Flip Y so origin is bottom-left and scale coordinates, with offset
        const plotX = OFFSET_X + pos.x * MULTIPLIER;
        const plotY = canvas.height - OFFSET_Y - (pos.y * MULTIPLIER);
        ctx.beginPath();
        ctx.arc(plotX, plotY, 10, 0, 2 * Math.PI);
        ctx.fillStyle = '#007bff';
        ctx.fill();
        ctx.strokeStyle = '#333';
        ctx.stroke();
        ctx.font = '14px Arial';
        ctx.fillStyle = '#333';
        ctx.fillText(pos.tag_id, plotX + 12, plotY);
    });
}

// --- RSSI Graphs ---
const rssiHistory = {
    scan1: [],
    scan2: [],
    scan3: [],
    scan4: []
};
const MAX_POINTS = 50; // Number of points to show in graph

function rollingAverage(arr, windowSize) {
    if (arr.length < windowSize) return [];
    const result = [];
    for (let i = 0; i <= arr.length - windowSize; i++) {
        const window = arr.slice(i, i + windowSize);
        const avg = window.reduce((a, b) => a + b, 0) / window.length;
        result.push(avg);
    }
    // Pad the start so filtered and raw arrays align visually
    while (result.length < arr.length) {
        result.unshift(result[0]);
    }
    return result;
}

function drawRssiGraph(scanKey, canvasId) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const data = rssiHistory[scanKey];
    // Draw original RSSI line
    ctx.strokeStyle = '#007bff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    if (data.length > 0) {
        for (let i = 0; i < data.length; i++) {
            const x = (i / (MAX_POINTS - 1)) * canvas.width;
            const y = canvas.height - ((data[i] + 100) * (canvas.height / 60));
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }
    // Draw filtered RSSI line (rolling average)
    const filtered = rollingAverage(data, 5); // window size 5
    ctx.strokeStyle = '#ff9900';
    ctx.lineWidth = 2;
    ctx.beginPath();
    if (filtered.length > 0) {
        for (let i = 0; i < filtered.length; i++) {
            const x = (i / (MAX_POINTS - 1)) * canvas.width;
            const y = canvas.height - ((filtered[i] + 100) * (canvas.height / 60));
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }
    // Draw axis
    ctx.strokeStyle = '#aaa';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, canvas.height - 1);
    ctx.lineTo(canvas.width, canvas.height - 1);
    ctx.stroke();
    ctx.font = '12px Arial';
    ctx.fillStyle = '#333';
    ctx.fillText('-100', 2, canvas.height - 2);
    ctx.fillText('-40', 2, 12);
    // Legend
    ctx.font = '12px Arial';
    ctx.fillStyle = '#007bff';
    ctx.fillText('Raw', canvas.width - 50, 20);
    ctx.fillStyle = '#ff9900';
    ctx.fillText('Avg', canvas.width - 50, 36);
}

function updateRssiGraphs() {
    drawRssiGraph('scan1', 'rssiScan1');
    drawRssiGraph('scan2', 'rssiScan2');
    drawRssiGraph('scan3', 'rssiScan3');
    drawRssiGraph('scan4', 'rssiScan4');
}

// Listen for RSSI feed
const FEED_PATH = '/feed/7C:EC:79:47:6C:5E';
db.ref(FEED_PATH).on('value', snapshot => {
    const feed = snapshot.val() || {};
    ['scan1', 'scan2', 'scan3', 'scan4'].forEach(scanKey => {
        if (feed[scanKey]) {
            rssiHistory[scanKey].push(feed[scanKey].rssi);
            if (rssiHistory[scanKey].length > MAX_POINTS) {
                rssiHistory[scanKey].shift();
            }
        }
    });
    updateRssiGraphs();
});
