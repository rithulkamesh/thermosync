package data

import (
	"log"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo/options"
)

type BodyTemp struct {
	Timestamp int     `bson:"timestamp,omitempty" json:"timestamp,omitempty"`
	BodyTemp  float32 `bson:"bodytemp,omitempty" json:"bodytemp,omitempty"`
	AcTemp    float32 `bson:"actemp,omitempty" json:"actemp,omitempty"`
}

func GetBodyTemp(timestamp int) (*BodyTemp, error) {
	filter := bson.M{"timestamp": timestamp}
	var result BodyTemp
	err := bodyTempColl.FindOne(ctx, filter).Decode(&result)

	if err != nil {
		return nil, err
	}
	return &result, nil
}

func (t *BodyTemp) Save() {
	filter := bson.M{"timestamp": t.Timestamp}
	update := bson.M{
		"$set": bson.M{
			"bodytemp": t.BodyTemp,
			"actemp":   t.AcTemp,
		},
	}

	opts := options.Update().SetUpsert(true)

	_, err := bodyTempColl.UpdateOne(
		ctx,
		filter,
		update,
		opts,
	)
	if err != nil {
		log.Fatal(err)
	}
}
