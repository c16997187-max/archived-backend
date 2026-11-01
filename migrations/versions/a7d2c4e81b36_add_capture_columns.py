"""Add phone capture columns

Revision ID: a7d2c4e81b36
Revises: 8c3e5d1a9f27
Create Date: 2026-11-01 22:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'a7d2c4e81b36'
down_revision: Union[str, Sequence[str], None] = '8c3e5d1a9f27'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('capture_sessions', sa.Column('phone_token_hash', sa.String(), nullable=True))
    op.add_column('capture_sessions', sa.Column('upload_count', sa.SmallInteger(), server_default=sa.text('0'), nullable=False))
    op.add_column('capture_sessions', sa.Column('duplicate_count', sa.SmallInteger(), server_default=sa.text('0'), nullable=False))
    op.create_unique_constraint(op.f('uq_capture_sessions_phone_token_hash'), 'capture_sessions', ['phone_token_hash'])
    op.create_check_constraint(op.f('ck_capture_sessions_upload_count_not_negative'), 'capture_sessions', 'upload_count >= 0')
    op.create_check_constraint(op.f('ck_capture_sessions_duplicate_count_not_negative'), 'capture_sessions', 'duplicate_count >= 0')
    op.add_column('documents', sa.Column('crop_corners', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.create_index(op.f('ix_documents_capture_session_id'), 'documents', ['capture_session_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_documents_capture_session_id'), table_name='documents')
    op.drop_column('documents', 'crop_corners')
    op.drop_constraint(op.f('ck_capture_sessions_duplicate_count_not_negative'), 'capture_sessions', type_='check')
    op.drop_constraint(op.f('ck_capture_sessions_upload_count_not_negative'), 'capture_sessions', type_='check')
    op.drop_constraint(op.f('uq_capture_sessions_phone_token_hash'), 'capture_sessions', type_='unique')
    op.drop_column('capture_sessions', 'duplicate_count')
    op.drop_column('capture_sessions', 'upload_count')
    op.drop_column('capture_sessions', 'phone_token_hash')
