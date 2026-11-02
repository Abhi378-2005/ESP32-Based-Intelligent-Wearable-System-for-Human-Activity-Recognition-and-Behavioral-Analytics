/*
 * PostureGuard Pro — Dual USB Serial + Wireless Bluetooth SPP Firmware
 * Board: ESP32 DevKit V1 | Sensor: MPU6050 (I2C: SDA=21, SCL=22, Addr=0x68)
 * Pure 3.3V Digital Telemetry Architecture
 * Streams simultaneously over USB Serial (115200 baud) AND Bluetooth SPP ("PostureGuard_Pro")
 */

#include <Wire.h>
#include <math.h>
#include "BluetoothSerial.h"

#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled! Please run `make menuconfig` to enable it
#endif

#define SDA_PIN 21
#define SCL_PIN 22
#define MPU_ADDR 0x68

BluetoothSerial SerialBT;

const float TILT_THRESHOLD = 25.0;            // Allowed posture tilt deviation (+/- 25.0 deg)
const unsigned long ALERT_DELAY_MS = 3000;    // 3.0s sustained slouch debounce timer
const unsigned long SAMPLE_INTERVAL_MS = 200; // 5Hz (200ms) telemetry update rate

float pitchOffset = 0.0;
float rollOffset = 0.0;
unsigned long tiltStartTime = 0;
unsigned long lastSampleTime = 0;
bool isTilted = false;

bool readIMU(float &ax_g, float &ay_g, float &az_g, float &gx_dps, float &gy_dps, float &gz_dps, float &pitch, float &roll) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) return false;

  Wire.requestFrom(MPU_ADDR, 14, true);
  if (Wire.available() == 14) {
    int16_t ax = (Wire.read() << 8) | Wire.read();
    int16_t ay = (Wire.read() << 8) | Wire.read();
    int16_t az = (Wire.read() << 8) | Wire.read();
    int16_t temp = (Wire.read() << 8) | Wire.read();
    int16_t gx = (Wire.read() << 8) | Wire.read();
    int16_t gy = (Wire.read() << 8) | Wire.read();
    int16_t gz = (Wire.read() << 8) | Wire.read();

    ax_g = ax / 16384.0;
    ay_g = ay / 16384.0;
    az_g = az / 16384.0;

    gx_dps = gx / 131.0;
    gy_dps = gy / 131.0;
    gz_dps = gz / 131.0;

    pitch = atan2(ax_g, sqrt(ay_g * ay_g + az_g * az_g)) * 180.0 / PI;
    roll  = atan2(ay_g, sqrt(ax_g * ax_g + az_g * az_g)) * 180.0 / PI;
    return true;
  }
  return false;
}

void setup() {
  Serial.begin(115200);
  SerialBT.begin("PostureGuard_Pro"); // Wireless Bluetooth SPP Device Name
  delay(500);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0x00);
  if (Wire.endTransmission() != 0) {
    Serial.println("ERROR: MPU6050 not detected!");
    SerialBT.println("ERROR: MPU6050 not detected!");
    while (1) delay(1000);
  }

  Serial.println("=========================================");
  Serial.println("CALIBRATING: Hold posture...");
  SerialBT.println("CALIBRATING: Hold posture...");

  float sumPitch = 0.0, sumRoll = 0.0;
  int validSamples = 0;
  for (int i = 0; i < 30; i++) {
    float ax, ay, az, gx, gy, gz, p, r;
    if (readIMU(ax, ay, az, gx, gy, gz, p, r)) {
      sumPitch += p;
      sumRoll += r;
      validSamples++;
    }
    delay(100);
  }

  if (validSamples > 0) {
    pitchOffset = sumPitch / validSamples;
    rollOffset  = sumRoll / validSamples;
  }

  String lockMsg = "Baseline Locked! Pitch Offset: " + String(pitchOffset, 1) +
                   " deg | Roll Offset: " + String(rollOffset, 1) + " deg";
  Serial.println(lockMsg);
  Serial.println("USB Serial + Bluetooth SPP ('PostureGuard_Pro') Streaming Active.");
  Serial.println("=========================================");

  SerialBT.println(lockMsg);
  SerialBT.println("Bluetooth SPP ('PostureGuard_Pro') Streaming Active.");
}

void loop() {
  if (millis() - lastSampleTime >= SAMPLE_INTERVAL_MS) {
    lastSampleTime = millis();

    float ax, ay, az, gx, gy, gz, rawPitch, rawRoll;
    if (!readIMU(ax, ay, az, gx, gy, gz, rawPitch, rawRoll)) return;

    float relPitch = rawPitch - pitchOffset;
    float relRoll  = rawRoll - rollOffset;
    float svm = sqrt(ax * ax + ay * ay + az * az);
    float gyroMag = sqrt(gx * gx + gy * gy + gz * gz);

    String statusMsg = "STATUS: [OK - Good Posture]";
    if (abs(relPitch) > TILT_THRESHOLD || abs(relRoll) > TILT_THRESHOLD) {
      if (!isTilted) {
        isTilted = true;
        tiltStartTime = millis();
      }
      unsigned long elapsed = millis() - tiltStartTime;
      if (elapsed >= ALERT_DELAY_MS) {
        statusMsg = "STATUS: [ALERT - BAD POSTURE!]";
      } else {
        float rem = (ALERT_DELAY_MS - elapsed) / 1000.0;
        statusMsg = "STATUS: [Tilted... " + String(rem, 1) + "s until alert]";
      }
    } else {
      isTilted = false;
    }

    // Unified Telemetry String for both USB Cable and Bluetooth SPP
    String telemetry = "Rel Pitch: " + String(relPitch, 1) +
                       " deg | Rel Roll: " + String(relRoll, 1) +
                       " deg | SVM: " + String(svm, 2) +
                       " g | Gyro: " + String(gyroMag, 1) +
                       " dps | " + statusMsg;

    Serial.println(telemetry);
    SerialBT.println(telemetry);
  }
}