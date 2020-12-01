"""create tweet_model table

Revision ID: 742425105201
Revises: 
Create Date: 2020-05-16 16:12:57.297415

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '742425105201'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'tweet_model',
        sa.Column('id', sa.BigInteger, primary_key=True, nullable=False),
        sa.Column('text', sa.String, nullable=False),
        sa.Column('retweet_count', sa.Integer, nullable=False),
        sa.Column('favorite_count', sa.Integer, nullable=False),
        sa.Column('user_name', sa.String, nullable=False),
        sa.Column('user_id', sa.BigInteger, nullable=False),
        sa.Column('parent_tweet_id', sa.BigInteger, sa.ForeignKey('tweet_model.id'), nullable=True)
    )


def downgrade():
    op.drop_table('tweet_model')
