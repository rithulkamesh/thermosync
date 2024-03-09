package data

import (
	"context"
	"log"
	"os"

	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

var ctx = context.TODO()
var mongoClient *mongo.Client
var bodyTempColl *mongo.Collection
var airQualityColl *mongo.Collection

// Initialize the MongoDB client and set up the required collections.
func InitDB() {
	var err error
	mongoClient, err = mongo.Connect(ctx, options.Client().ApplyURI(os.Getenv("MONGO_URI")))
	if err != nil {
		log.Fatal(err)
	}
	initCollections()
}

func initCollections() {
	bodyTempColl = mongoClient.Database("thermosync").Collection("bodyTempData")
	airQualityColl = mongoClient.Database("thermosync").Collection("airQualityData")
}
