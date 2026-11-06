"""Add OCR path columns

Revision ID: c91e4a7d2f58
Revises: a7d2c4e81b36
Create Date: 2026-11-05 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'c91e4a7d2f58'
down_revision: Union[str, Sequence[str], None] = 'a7d2c4e81b36'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('extractions', sa.Column('ocr_path', sa.String(), nullable=True))
    op.add_column('extractions', sa.Column('template_id', sa.BigInteger(), nullable=True))
    op.add_column('extractions', sa.Column('alignment_score', sa.Float(), nullable=True))
    op.add_column('extractions', sa.Column('full_text', sa.String(), nullable=True))
    op.create_foreign_key(op.f('fk_extractions_template_id_form_templates'), 'extractions', 'form_templates', ['template_id'], ['id'])
    op.create_check_constraint(op.f('ck_extractions_ocr_path'), 'extractions', "ocr_path IN ('template', 'generic')")
    op.add_column('ocr_results', sa.Column('needs_check', sa.Boolean(), server_default=sa.text('false'), nullable=False))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('ocr_results', 'needs_check')
    op.drop_constraint(op.f('ck_extractions_ocr_path'), 'extractions', type_='check')
    op.drop_constraint(op.f('fk_extractions_template_id_form_templates'), 'extractions', type_='foreignkey')
    op.drop_column('extractions', 'full_text')
    op.drop_column('extractions', 'alignment_score')
    op.drop_column('extractions', 'template_id')
    op.drop_column('extractions', 'ocr_path')
