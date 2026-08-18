import boto3
import json
import os


def main():
    comprehend = boto3.client("comprehend")
    s3 = boto3.client("s3")
    bucket = "gakken"

    obj = s3.get_object(Bucket=bucket, Key="case_study.txt")
    text = obj["Body"].read().decode("utf-8")

    entities = comprehend.detect_entities(Text=text, LanguageCode="en")["Entities"]

    sentiment = comprehend.detect_sentiment(Text=text, LanguageCode="en")

    phrases = comprehend.detect_key_phrases(Text=text, LanguageCode="en")["KeyPhrases"]

    scores = sentiment["SentimentScore"]
    entities_by_type = {}
    for e in entities:
        entities_by_type.setdefault(e["Type"], [])
        if e["Text"] not in entities_by_type[e["Type"]]:
            entities_by_type[e["Type"]].append(e["Text"])

    metadata = {
        "metadataAttributes": {
            "sentiment": sentiment["Sentiment"],
            "sentiment_positive_score": round(scores["Positive"], 3),
            "sentiment_negative_score": round(scores["Negative"], 3),
            "organizations": entities_by_type.get("ORGANIZATION", []),
            "people": entities_by_type.get("PERSON", []),
            "locations": entities_by_type.get("LOCATION", []),
            "dates": entities_by_type.get("DATE", []),
            "key_phrases": [
                p["Text"]
                for p in sorted(phrases, key=lambda x: x["Score"], reverse=True)[:10]
            ],
        }
    }

    s3.put_object(
        Bucket=bucket,
        Key="case_study.txt.metadata.json",
        Body=json.dumps(metadata, indent=2),
        ContentType="application/json",
    )

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
