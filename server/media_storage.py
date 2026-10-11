"""Private image storage: Neon Object Storage's S3 API, or local folders when Neon isn't configured."""
import os
from pathlib import PurePosixPath

import cv2
import numpy as np

from database_config import load_database_env, use_local_storage

MEDIA_KINDS = frozenset(("crops", "frames", "keyframes"))


class MediaNotFound(Exception):
    pass


class MediaStorage:
    def __init__(self, bucket=None):
        load_database_env()
        required = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_ENDPOINT_URL_S3", "AWS_REGION")
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            raise RuntimeError(
                "Neon Object Storage configuration is missing: " + ", ".join(missing)
                + ". Run `npx -y neon@latest env --write .env.local`."
            )
        import boto3
        from botocore.config import Config

        self.bucket = bucket or os.environ.get("BEACON_MEDIA_BUCKET", "recall-images")
        self.client = boto3.client(
            "s3",
            endpoint_url=os.environ["AWS_ENDPOINT_URL_S3"],
            region_name=os.environ["AWS_REGION"],
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )

    @staticmethod
    def key(path):
        """Convert local-style legacy paths into stable bucket keys."""
        parts = [part for part in str(path).replace("\\", "/").split("/") if part]
        if len(parts) < 2 or parts[-2] not in MEDIA_KINDS:
            raise ValueError(f"Unsupported media path: {path}")
        name = parts[-1]
        if name in (".", "..") or "/" in name or "\\" in name:
            raise ValueError(f"Unsafe media path: {path}")
        return str(PurePosixPath(parts[-2]) / name)

    def put_bytes(self, path, data, content_type="image/jpeg"):
        self.client.put_object(
            Bucket=self.bucket,
            Key=self.key(path),
            Body=data,
            ContentType=content_type,
            CacheControl="private, max-age=3600",
        )
        return path

    def put_image(self, path, image, quality=90):
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise ValueError("Could not encode image as JPEG")
        return self.put_bytes(path, encoded.tobytes())

    def get_bytes(self, path):
        from botocore.exceptions import ClientError

        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self.key(path))
            return response["Body"].read()
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey", "NotFound"):
                raise MediaNotFound(path) from exc
            raise

    def get_image(self, path):
        data = np.frombuffer(self.get_bytes(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Stored object is not a valid image: {self.key(path)}")
        return image

    def delete(self, path):
        if path:
            self.client.delete_object(Bucket=self.bucket, Key=self.key(path))

    def list_keys(self):
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket):
            for item in page.get("Contents", ()):
                yield item["Key"]

    def delete_unreferenced(self, referenced_paths):
        keep = {self.key(path) for path in referenced_paths if path}
        stale = [key for key in self.list_keys() if key not in keep]
        for start in range(0, len(stale), 1000):
            self.client.delete_objects(
                Bucket=self.bucket,
                Delete={"Objects": [{"Key": key} for key in stale[start:start + 1000]], "Quiet": True},
            )
        return len(stale)

    def clear(self):
        return self.delete_unreferenced(())


class LocalMediaStorage(MediaStorage):
    """The same interface, backed by <data>/crops, <data>/frames and <data>/keyframes on this machine."""

    def __init__(self, data_dir="data"):
        self.root = data_dir
        for kind in MEDIA_KINDS:
            os.makedirs(os.path.join(data_dir, kind), exist_ok=True)

    def _file(self, path):
        return os.path.join(self.root, *self.key(path).split("/"))

    def put_bytes(self, path, data, content_type="image/jpeg"):
        tmp = self._file(path) + ".part"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, self._file(path))
        return path

    def get_bytes(self, path):
        try:
            with open(self._file(path), "rb") as fh:
                return fh.read()
        except FileNotFoundError as exc:
            raise MediaNotFound(path) from exc

    def delete(self, path):
        if path:
            try:
                os.remove(self._file(path))
            except FileNotFoundError:
                pass

    def list_keys(self):
        for kind in sorted(MEDIA_KINDS):
            folder = os.path.join(self.root, kind)
            for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else ():
                if os.path.isfile(os.path.join(folder, name)) and not name.endswith(".part"):
                    yield f"{kind}/{name}"

    def delete_unreferenced(self, referenced_paths):
        keep = {self.key(path) for path in referenced_paths if path}
        stale = [key for key in self.list_keys() if key not in keep]
        for key in stale:
            self.delete(key)
        return len(stale)


def open_media(data_dir="data"):
    """Neon Object Storage when Neon is configured, otherwise local folders under data_dir."""
    return LocalMediaStorage(data_dir) if use_local_storage() else MediaStorage()
