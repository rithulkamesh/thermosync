package main

import (
	"context"
	"net/http"
	"os"

	"github.com/labstack/echo"
	"github.com/labstack/echo/middleware"
	"github.com/rithulkamesh/thermosync/data"
	"go.mongodb.org/mongo-driver/mongo"
)

var collection *mongo.Collection
var ctx = context.TODO()
var lastAcTemp = 23

func main() {
	e := echo.New()

	data.InitDB()
	e.Use(middleware.CORSWithConfig(middleware.CORSConfig{
		AllowHeaders: []string{"Authorization", "content-type"},
		AllowOrigins: []string{"*"},
		AllowMethods: []string{http.MethodPost, http.MethodDelete},
	}))

	e.POST("/airquality", handleAirQuality)
	e.POST("/bodytemp", handleBodyTemp)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8000"
	}
	e.Logger.Fatal(e.Start("0.0.0.0:" + port))
}

func handleAirQuality(c echo.Context) error {
	var aq data.AirQualityData
	if err := c.Bind(&aq); err != nil {
		return err
	}

	if aq.Temp == 0 || aq.Humidity == 0 || aq.Co2 == 0 {
		return c.JSON(http.StatusBadRequest, "missing fields")
	}

	aq.Save()
	return c.String(http.StatusOK, "OK")
}

func handleBodyTemp(c echo.Context) error {
	var bt data.BodyTemp
	if err := c.Bind(&bt); err != nil {
		return err
	}

	if bt.BodyTemp == 0 {
		return c.JSON(http.StatusBadRequest, "missing fields")
	}

	bt.AcTemp = lastAcTemp

	bt.Save()
	return c.String(http.StatusOK, "OK")
}
