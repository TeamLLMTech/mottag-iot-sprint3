#include <Arduino.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <BLEBeacon.h>

#define DEVICE_NAME         "ESP32"
#define SERVICE_UUID        "7A0247E7-8E88-409B-A959-AB5092DDB03E"
#define BEACON_UUID         "2D7A9F0C-E0E8-4CC9-A71B-A21DB2D034A1"
#define BEACON_UUID_REV     "A134D0B2-1DA2-1BA7-C94C-E8E00C9F7A2D"
#define CHARACTERISTIC_UUID "82258BAA-DF72-47E8-99BC-B73D7ECD08A5"

BLEServer *pServer;
BLECharacteristic *pCharacteristic;
bool deviceConnected = false;
uint8_t value = 0;

int buzzer1 = 25; // GPIO conectado ao buzzer
int ledPin = 18; // GPIO conectado ao LED

bool shouldMakeAnnouncement = false;

void playTone(int frequency, int duration) {
  if (frequency > 0) {
    ledcWriteTone(0, frequency);
    ledcWrite(0, 255);
  } else {
    ledcWriteTone(0, 0);
  }
  delay(duration);
  ledcWriteTone(0, 0);
}

void playAnnounceTone() {
  int melody[] = { 1000, 1200, 1500, 1800, 2000 };
  int noteDurations[] = { 200, 200, 200, 200, 400 };

  for (int i = 0; i < 5; i++) {
    playTone(melody[i], noteDurations[i]);
    delay(50);
  }
}

void makeAnnouncement() {
  digitalWrite(ledPin, HIGH);
  playAnnounceTone();
  digitalWrite(ledPin, LOW);
  delay(1000);
}

class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *pServer) {
    deviceConnected = true;
    Serial.println("deviceConnected = true");
  };

  void onDisconnect(BLEServer *pServer) {
    deviceConnected = false;
    Serial.println("deviceConnected = false");

    BLEAdvertising *pAdvertising;
    pAdvertising = pServer->getAdvertising();
    pAdvertising->start();
    Serial.println("iBeacon advertising reiniciado");
  }
};

class MyCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *pCharacteristic) {
    String rxValue = String(pCharacteristic->getValue().c_str());

    if (rxValue.length() > 0) {
      Serial.println("*********");
      Serial.print("Valor recebido: ");
      for (int i = 0; i < rxValue.length(); i++) {
        Serial.print(rxValue[i]);
      }
      Serial.println();
      Serial.println("*********");

      shouldMakeAnnouncement = !shouldMakeAnnouncement;
    }
  }
};

void init_service() {
  BLEAdvertising *pAdvertising;
  pAdvertising = pServer->getAdvertising();
  pAdvertising->stop();

  BLEService *pService = pServer->createService(BLEUUID(SERVICE_UUID));

  pCharacteristic = pService->createCharacteristic(
    CHARACTERISTIC_UUID, BLECharacteristic::PROPERTY_WRITE
  );
  pCharacteristic->setCallbacks(new MyCallbacks());
  pCharacteristic->addDescriptor(new BLE2902());

  pAdvertising->addServiceUUID(BLEUUID(SERVICE_UUID));

  pService->start();

  pAdvertising->start();
}

void init_beacon() {
  BLEAdvertising *pAdvertising;
  pAdvertising = pServer->getAdvertising();
  pAdvertising->stop();

  BLEBeacon myBeacon;
  myBeacon.setManufacturerId(0x4c00);
  myBeacon.setMajor(5);
  myBeacon.setMinor(88);
  myBeacon.setSignalPower(0xc5);
  myBeacon.setProximityUUID(BLEUUID(BEACON_UUID_REV));

  BLEAdvertisementData advertisementData;
  advertisementData.setFlags(0x1A);
  advertisementData.setManufacturerData(myBeacon.getData());
  pAdvertising->setAdvertisementData(advertisementData);

  pAdvertising->start();
}

void setup() {
  Serial.begin(115200);
  Serial.println();
  Serial.println("Iniciando...");
  Serial.flush();

  BLEDevice::init(DEVICE_NAME);
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  init_service();
  init_beacon();

  pinMode(ledPin, OUTPUT);
  ledcSetup(0, 2000, 8);
  ledcAttachPin(buzzer1, 0);

  Serial.println("iBeacon + service definidos e anunciando!");
}

void loop() {
  if (shouldMakeAnnouncement) {
    makeAnnouncement();
  }
}
