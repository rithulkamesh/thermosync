#include <DHT11.h>
#include <MQ135.h>
#include "ThingSpeak.h"
#include <ESP8266WiFi.h>
#include <Adafruit_SH110X.h>
#include <Wire.h>

// Define constants
#define I2C_ADDRESS 0x3c
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define CO2_ZERO 55
#define DHT_PIN 2
#define CO2_PIN A0
#define CHANNEL_NUMBER 2463215

#define WRITE_API_KEY "KCPK8E9WC0KOSGRP" // Invalidated
#define SSID "Epsilon" // Mobile hotspot to handle local network
#define PASSWORD "d56a666q"  // Invalidated

Adafruit_SH1106G display = Adafruit_SH1106G(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
DHT11 dht11(DHT_PIN);
WiFiClient client;

void setup() {
    Serial.begin(9600);
    display.begin(I2C_ADDRESS, true);
    dht11.setDelay(500);

    WiFi.mode(WIFI_STA);
    ThingSpeak.begin(client);
}

void loop() {
    // Connect to Wi-Fi if not connected
    connectToWiFi();

    // Read temperature and humidity
    int temperature, humidity;
    readDHT11(&temperature, &humidity);

    // Read CO2 levels
    int co2ppm = readCO2Levels();

    // Display readings on OLED
    displayReadings(temperature, humidity, co2ppm);

    // Send data to ThingSpeak
    sendDataToThingSpeak(temperature, humidity, co2ppm);

    delay(20000);
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

void readDHT11(int* temperature, int* humidity) {
    int result = dht11.readTemperatureHumidity(*temperature, *humidity);
    if (result != 0) {
        Serial.println(DHT11::getErrorString(result));
    }
}

int readCO2Levels() {
    int co2now[10];
    int co2raw = 0;
    int co2comp = 0;
    int co2ppm = 0;
    int sum = 0;

    for (int i = 0; i < 10; i++) {
        co2now[i] = analogRead(CO2_PIN);
        delay(200);
        sum += co2now[i];
    }

    co2raw = sum / 10;
    co2comp = co2raw - CO2_ZERO;
    co2ppm = map(co2comp, 0, 1023, 400, 5000);

    return co2ppm;
}

void displayReadings(int temperature, int humidity, int co2ppm) {
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SH110X_WHITE);
    display.setCursor(0, 0);
    display.print("CO2 Levels: ");
    display.print(co2ppm);
    display.print("ppm\n");
    display.print("Temperature: ");
    display.print(temperature);
    display.print(" C\n");
    display.print("Humidity: ");
    display.print(humidity);
    display.print("%\n");
    display.display();
}

void sendDataToThingSpeak(int temperature, int humidity, int co2ppm) {
    ThingSpeak.setField(1, temperature);
    ThingSpeak.setField(2, humidity);
    ThingSpeak.setField(3, co2ppm);

    int httpCode = ThingSpeak.writeFields(CHANNEL_NUMBER, WRITE_API_KEY);
    if (httpCode == 200) {
        Serial.println("Channel write successful.");
    } else {
        Serial.println("Problem writing to channel. HTTP error code " + String(httpCode));
    }
}
