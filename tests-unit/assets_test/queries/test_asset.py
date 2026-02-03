from sqlalchemy.orm import Session

from app.assets.database.models import Asset
from app.assets.database.queries import asset_exists_by_hash, get_asset_by_hash


class TestAssetExistsByHash:
    def test_returns_false_for_nonexistent(self, session: Session):
        assert asset_exists_by_hash(session, asset_hash="nonexistent") is False

    def test_returns_true_for_existing(self, session: Session):
        asset = Asset(hash="blake3:abc123", size_bytes=100)
        session.add(asset)
        session.commit()

        assert asset_exists_by_hash(session, asset_hash="blake3:abc123") is True

    def test_does_not_match_null_hash(self, session: Session):
        asset = Asset(hash=None, size_bytes=100)
        session.add(asset)
        session.commit()

        assert asset_exists_by_hash(session, asset_hash="") is False


class TestGetAssetByHash:
    def test_returns_none_for_nonexistent(self, session: Session):
        assert get_asset_by_hash(session, asset_hash="nonexistent") is None

    def test_returns_asset_for_existing(self, session: Session):
        asset = Asset(hash="blake3:def456", size_bytes=200, mime_type="image/png")
        session.add(asset)
        session.commit()

        result = get_asset_by_hash(session, asset_hash="blake3:def456")
        assert result is not None
        assert result.id == asset.id
        assert result.size_bytes == 200
        assert result.mime_type == "image/png"
