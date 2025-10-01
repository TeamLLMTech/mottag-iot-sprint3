#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEScan.h>
#include <BLEAdvertisedDevice.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <env.h>
#include <vector>
#include <esp_bt.h>
#include <esp_coexist.h>

// ---------------- INFO ----------------
// ---------------- TIMER ---------------
unsigned long lastScanTime = 0;
const unsigned long scanIntervalMs = 3000;   // janela de varredura + envio (3 s)
const int scanTimeSec = scanIntervalMs / 1000;
// --------------- WIFI -----------------
void setupWifi () {
  WiFi.mode(WIFI_STA);        // força STA, evita AP+STA desnecessário
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) { Serial.print("."); delay(300); }
  Serial.println();
}

// --------------- HTTP -----------------
HTTPClient http;
// --------------- BLE ------------------
// Endereços conhecidos
const char* knownDevices[] = {
  "7C:EC:79:47:6C:5E",
  "7C:EC:79:47:89:BB",
  "D4:F5:13:79:E2:39",
  "51:00:24:06:00:FD",
};
const int knownDeviceCount = sizeof(knownDevices) / sizeof(knownDevices[0]);

BLEScan *pBLEScan;

struct Sample {
  String addr;
  int rssi;
  uint32_t t_ms;
};
std::vector<Sample> batch;  // buffer de eventos desta janela

static inline bool isKnown(const String& addr) {
  for (int i = 0; i < knownDeviceCount; i++)
    if (addr.equalsIgnoreCase(knownDevices[i])) return true;
  return false;
}

class MyAdvertisedDeviceCallbacks : public BLEAdvertisedDeviceCallbacks {
  void onResult(BLEAdvertisedDevice advertisedDevice) override {
    String addr = advertisedDevice.getAddress().toString().c_str();
    if (!isKnown(addr)) return;

    int rssi = advertisedDevice.getRSSI();
    if (rssi > -35) return; // ignora saturados muito próximos

    // Coleta crua no buffer; enviamos em lote ao final da varredura
    Sample s{addr, rssi, millis()};
    batch.push_back(s);
  }
};

void setupBle () {
  esp_bt_controller_mem_release(ESP_BT_MODE_CLASSIC_BT);  // só BLE
  esp_coex_preference_set(ESP_COEX_PREFER_BALANCE);
  BLEDevice::init("");
  pBLEScan = BLEDevice::getScan();
  pBLEScan->setAdvertisedDeviceCallbacks(new MyAdvertisedDeviceCallbacks());

  // Para medir RSSI, prefira varredura PASSIVA
  pBLEScan->setActiveScan(false);

  // Quase contínuo: janela ≈ intervalo (ms)
  pBLEScan->setInterval(120);
  pBLEScan->setWindow(120);
}

// Faz uma varredura por scanTimeSec e, ao término, envia o lote coletado
void scanAndFlush () {
  Serial.println("[BLE] Scanning...");
  pBLEScan->start(scanTimeSec, false); // bloqueia por scanTimeSec
  Serial.printf("[BLE] Scan done! Batch size: %u\n", (unsigned)batch.size());

  if (batch.empty()) return;

  // Monta JSON: { aid, time, events: [ {addr,rssi,t}, ... ] }
  String json = "{\"aid\":\"" + String(ANTENA_ID) + "\",\"time\":" + String(millis()) + ",\"events\":[";
  for (size_t i = 0; i < batch.size(); ++i) {
    const auto &e = batch[i];
    json += "{\"addr\":\"" + e.addr + "\",\"rssi\":" + String(e.rssi) + ",\"t\":" + String(e.t_ms) + "}";
    if (i + 1 < batch.size()) json += ",";
  }
  json += "]}";

  http.setReuse(true); // tenta manter conexão viva
  http.begin(String(API_BASE_URL));
  http.addHeader("Content-Type", "application/json");
  http.addHeader("x-api-key", API_KEY);

  int httpCode = http.POST(json);
  if (httpCode > 0) {
    Serial.printf("[HTTP] POST batch sent, code: %d, bytes: %u\n", httpCode, json.length());
  } else {
    Serial.printf("[HTTP] POST failed, err: %s\n", http.errorToString(httpCode).c_str());
  }
  http.end();
  batch.clear();

  // Limpa resultados do scanner para liberar memória
  pBLEScan->clearResults();
}

void setup() {
  Serial.begin(115200);
  setupWifi();
  setupBle();
}

void loop() {
  const unsigned long now = millis();
  if (now - lastScanTime >= scanIntervalMs) {
    lastScanTime = now;
    scanAndFlush();
  }
}
