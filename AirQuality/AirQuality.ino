#include <DHT11.h>
#include <MQ135.h>
#include "ThingSpeak.h"
#include <ESP8266WiFi.h>
#include <Adafruit_SH110X.h>

#define i2c_Address 0x3c
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64 
#define OLED_RESET -1 

#define anInput     A0
#define digTrigger   2
#define co2Zero     55   

Adafruit_SH1106G display = Adafruit_SH1106G(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
DHT11 dht11(2);

unsigned long channelNumber = 2463215;
const char * writeApiKey = "KCPK8E9WC0KOSGRP";


String ssid = "Epsilon";
String password = "d56a666q";
WiFiClient  client;

void setup() {
    Serial.begin(9600);
    display.begin(i2c_Address, true);
    dht11.setDelay(500);

     WiFi.mode(WIFI_STA);
     ThingSpeak.begin(client);
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

    int temperature = 0;
    int humidity = 0;
    
    int co2now[10];
    int co2raw = 0;
    int co2comp = 0;
    int co2ppm = 0;
    int zzz = 0;

    int result = dht11.readTemperatureHumidity(temperature, humidity);


    for (int x = 0;x<10;x++) {
      co2now[x]=analogRead(A0);
      delay(200);
    }

    for (int x = 0;x<10;x++){ 
      zzz=zzz + co2now[x];
    }

    co2raw = zzz/10;
    co2comp = co2raw - co2Zero;
    co2ppm = map(co2comp,0,1023,400,5000); 




    if (result == 0) {

    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SH110X_WHITE);
    display.setCursor(0, 0);
    display.print("C02 Levels: ");
    display.print(co2ppm);

    display.print("ppm\n");

    display.print("Temperature: ");
    display.print(temperature);
    display.print(" C\n");
    display.print("Humidity: ");

    display.print(humidity);
    display.print("%\n");
    display.display();

    ThingSpeak.setField(1, temperature);
    ThingSpeak.setField(2, humidity);
    ThingSpeak.setField(3, co2ppm);

  int httpCode = ThingSpeak.writeFields(channelNumber, writeApiKey);
  if (httpCode == 200) {
    Serial.println("Channel write successful.");
  }
  else {
    Serial.println("Problem writing to channel. HTTP error code " + String(httpCode));
  }

  delay(20000);
    } else {
        Serial.println(DHT11::getErrorString(result));
    }



}

