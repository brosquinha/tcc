"""add source field

Revision ID: 187bc3749acc
Revises: 742425105201
Create Date: 2020-07-13 11:48:25.986782

"""
import os

from alembic import op
import sqlalchemy as sa

from dataset.model.tweet_model import TweetModel
from collect_fakenewsnet_tweets import FakeNewsNetCollector, filelist


# revision identifiers, used by Alembic.
revision = '187bc3749acc'
down_revision = '742425105201'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('tweet_model', sa.Column('source', sa.String))

    collector = FakeNewsNetCollector([os.path.join('..', filelist[0])])
    politifact_ids = [int(item) for sublist in list(collector.get_tweets_ids_from_csv()) for item in sublist if item != '']

    connection = op.get_bind()
    connection.execute(
        sa.update(TweetModel).values(source="gossipcop")
    )
    connection.execute(
        sa.update(TweetModel).where(TweetModel.id.in_(politifact_ids)).values(source="politifact"),
    )


def downgrade():
    op.drop_column('tweet_model', 'source')
