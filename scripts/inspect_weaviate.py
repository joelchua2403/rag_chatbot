import json
from app.store import connect
if __name__ == "__main__":
    client = connect()
    try:
        for name in ["PaperVersion", "PaperChunk"]:
            coll = client.collections.get(name)
            print(f"{name}: {len(coll)} objects")
            for obj in coll.query.fetch_objects(limit=5).objects:
                props = dict(obj.properties)
                props.pop("full_text", None)
                if "text" in props:
                    props["text"] = props["text"][:150]
                print(json.dumps(props, ensure_ascii=False, default=str))
    finally:
        client.close()
