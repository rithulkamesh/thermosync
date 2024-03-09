package data

import (
	"log"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo/options"
)

type AirQualityData struct {
	Timestamp int     `bson:"timestamp,omitempty" json:"timestamp,omitempty"`
	Co2       float32 `bson:"co2,omitempty" json:"co2,omitempty"`
	Humidity  float32 `bson:"humidity,omitempty" json:"humidity,omitempty"`
	Temp      float32 `bson:"temp,omitempty" json:"temp,omitempty"`
}

func GetAirQualityData(timestamp int) (*AirQualityData, error) {
	filter := bson.M{"timestamp": timestamp}
	var result AirQualityData

	err := airQualityColl.FindOne(ctx, filter).Decode(&result)
	if err != nil {
		return nil, err
	}
	return &result, nil
}

func (a *AirQualityData) Save() {
	filter := bson.M{"timestamp": a.Timestamp}
	update := bson.M{
		"$set": bson.M{
			"co2":      a.Co2,
			"humidity": a.Humidity,
			"temp":     a.Temp,
		},
	}

	opts := options.Update().SetUpsert(true)

	_, err := airQualityColl.UpdateOne(
		ctx,
		filter,
		update,
		opts,
	)
	if err != nil {
		log.Fatal(err)
	}
}
