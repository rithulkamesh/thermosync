#include <WiFi.h>
#include <Adafruit_SH110X.h>
#include <HTTPClient.h>

#define i2c_Address 0x3c
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64 
#define OLED_RESET -1 


Adafruit_SH1106G display = Adafruit_SH1106G(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

String ssid = "Epsilon";
String password = "d56a666q";
WiFiClient  client;

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
    int httpCode = http.POST("{\"body_temp\":32}");

    if (httpCode > 0) {
        String payload = http.getString();
        Serial.println(payload);
        display.clearDisplay();
        display.setTextSize(1);
        display.setTextColor(SH110X_WHITE);
        display.setCursor(0, 0);
        display.print("AC Temp: ");
        display.println(payload);
    } else {
        Serial.printf("HTTP GET failed, error: %s\n", http.errorToString(httpCode).c_str());
    }

    http.end();
    delay(5000);

}
