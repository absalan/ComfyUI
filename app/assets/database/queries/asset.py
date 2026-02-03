import sqlalchemy as sa
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.assets.database.models import Asset


def asset_exists_by_hash(
    session: Session,
    *,
    asset_hash: str,
) -> bool:
    """
    Check if an asset with a given hash exists in database.
    """
    row = (
        session.execute(
            select(sa.literal(True)).select_from(Asset).where(Asset.hash == asset_hash).limit(1)
        )
    ).first()
    return row is not None


def get_asset_by_hash(
    session: Session,
    *,
    asset_hash: str,
) -> Asset | None:
    return (
        session.execute(select(Asset).where(Asset.hash == asset_hash).limit(1))
    ).scalars().first()
