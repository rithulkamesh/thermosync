#include <WiFi.h>
#include <Adafruit_SH110X.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

#define i2c_Address 0x3c
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1

Adafruit_SH1106G display = Adafruit_SH1106G(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

String ssid = "Epsilon";
String password = "d56a666q";
WiFiClient client;

bool compressorState;

void setup() {
    Serial.begin(9600);
    display.begin(i2c_Address, true);
    WiFi.mode(WIFI_STA);
}

void loop() {
    if (WiFi.status() != WL_CONNECTED) {
        Serial.print("Attempting to connect to SSID: ");
        Serial.println(ssid);
        while (WiFi.status() != WL_CONNECTED) {
            WiFi.begin(ssid, password);
            Serial.print(".");
            delay(5000);
        }
        Serial.println("\nConnected.");
    }

    HTTPClient http;
    http.begin(client, "http://164.52.194.247:5000/predict");

    http.addHeader("Content-Type", "application/json");

    StaticJsonDocument<200> doc;
    doc["body_temp"] = 37.5; // Replace with your actual value

    String requestBody;
    serializeJson(doc, requestBody);

    int httpCode = http.POST(requestBody);

    if (httpCode > 0) {
        String payload = http.getString();

        StaticJsonDocument<200> doc;
        deserializeJson(doc, payload);

        if (doc.containsKey("ac_temp")) {
            display.clearDisplay();
            display.setTextSize(1);
            display.setTextColor(SH110X_WHITE);
            display.setCursor(0, 0);
            display.print("AC Temp: ");
            display.println(doc["ac_temp"].as<float>());
            display.display();
        } else {
            Serial.println("Error: 'ac_temp' not found in response");
        }
    } else {
        Serial.printf("HTTP POST failed, error: %s\n", http.errorToString(httpCode).c_str());
    }

    http.end();
    delay(5000);
}