"""Add grade_level to enrollments

A draft may have a grade but no section yet, so the grade cannot live only on
the section.

Revision ID: 4b1f9c2e7a10
Revises: 07fdaac7a3d0
Create Date: 2026-11-01 18:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '4b1f9c2e7a10'
down_revision: Union[str, Sequence[str], None] = '07fdaac7a3d0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('enrollments', sa.Column('grade_level', sa.SmallInteger(), nullable=True))
    op.create_check_constraint(
        op.f('ck_enrollments_grade_level_k_to_12'), 'enrollments', 'grade_level BETWEEN 0 AND 12'
    )
    op.execute(
        "UPDATE enrollments SET grade_level = sections.grade_level "
        "FROM sections WHERE enrollments.section_id = sections.id"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(op.f('ck_enrollments_grade_level_k_to_12'), 'enrollments', type_='check')
    op.drop_column('enrollments', 'grade_level')
