#include <WiFi.h>
#include <Adafruit_SH110X.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <Wire.h>

// Define constants
#define I2C_ADDRESS 0x3c
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define SSID "Epsilon" // Mobile hotspot for local network
#define PASSWORD "d56a666q" // Invalidated
#define API_URL "http://localhost:5000/predict" // Offline

// Initialize objects
Adafruit_SH1106G display = Adafruit_SH1106G(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
WiFiClient client;

void setup() {
    Serial.begin(9600);
    display.begin(I2C_ADDRESS, true);
    WiFi.mode(WIFI_STA);
}

void loop() {
    // Connect to Wi-Fi if not connected
    connectToWiFi();

    // Send POST request to the API
    float acTemp = sendHttpRequest();

    // Display AC temperature on the OLED
    displayAcTemp(acTemp);

    delay(5000);
}

void connectToWiFi() {
    if (WiFi.status() != WL_CONNECTED) {
        Serial.print("Attempting to connect to SSID: ");
        Serial.println(SSID);
        while (WiFi.status() != WL_CONNECTED) {
            WiFi.begin(SSID, PASSWORD);
            Serial.print("Connecting...");
            delay(5000);
        }
        Serial.println("\nConnected.");
    }
}

float sendHttpRequest() {
    HTTPClient http;
    http.begin(client, API_URL);
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
            float acTemp = doc["ac_temp"].as<float>();
            http.end();
            return acTemp;
        } else {
            Serial.println("Error: 'ac_temp' not found in response");
        }
    } else {
        Serial.printf("HTTP POST failed, error: %s\n", http.errorToString(httpCode).c_str());
    }

    http.end();
    return 0.0; // Return a default value if the request fails
}

void displayAcTemp(float acTemp) {
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SH110X_WHITE);
    display.setCursor(0, 0);
    display.print("AC Temp: ");
    display.println(acTemp);
    display.display();
}
