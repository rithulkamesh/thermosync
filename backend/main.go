package main

import (
	"net/http"
	"os"
	"time"

	"github.com/labstack/echo"
	"github.com/labstack/echo/middleware"
)

type DataStruct struct {
	Timestamp int     `json:"timestamp"`
	BodyTemp  float32 `json:"bodytemp"`
	AcTemp    float32 `json:"actemp"`
}

func main() {
	e := echo.New()

	e.Use(middleware.CORSWithConfig(middleware.CORSConfig{
		AllowHeaders: []string{"Authorization", "content-type"},
		AllowOrigins: []string{"*"},
		AllowMethods: []string{http.MethodPost, http.MethodDelete},
	}))

	e.POST("/bodytemp", handleBodyTemp)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8000"
	}
	e.Logger.Fatal(e.Start("0.0.0.0:" + port))
}

func handleBodyTemp(c echo.Context) error {
	data := new(DataStruct)
	if err := c.Bind(data); err != nil {
		return err
	}

	t := time.Unix(int64(data.Timestamp), 0)
	c.Logger().Infof("Received data at %v: Body Temp = %.2f°C", t, data.BodyTemp)

	return c.JSON(http.StatusOK, data)
}
