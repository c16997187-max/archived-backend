"""Make learner name search accent-insensitive

Replaces the name trigram indexes with ones on the unaccented name, so "Pena"
finds "Peña" and the indexes still serve the query.

Revision ID: 8c3e5d1a9f27
Revises: 4b1f9c2e7a10
Create Date: 2026-11-01 20:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '8c3e5d1a9f27'
down_revision: Union[str, Sequence[str], None] = '4b1f9c2e7a10'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS unaccent SCHEMA public")
    # unaccent() is only STABLE, because its dictionary can change, so it cannot be
    # used in an index. We never change the dictionary; if that ever happens, reindex.
    # Schema-qualified because pg_restore rebuilds indexes with an empty search_path.
    op.execute(
        """
        CREATE FUNCTION public.immutable_unaccent(text) RETURNS text
        LANGUAGE sql IMMUTABLE PARALLEL SAFE STRICT
        AS $$ SELECT public.unaccent('public.unaccent'::regdictionary, $1) $$
        """
    )
    op.drop_index('ix_learners_last_name_trgm', table_name='learners')
    op.drop_index('ix_learners_first_name_trgm', table_name='learners')
    op.execute(
        "CREATE INDEX ix_learners_last_name_unaccent_trgm ON learners "
        "USING gin (public.immutable_unaccent(last_name) gin_trgm_ops)"
    )
    op.execute(
        "CREATE INDEX ix_learners_first_name_unaccent_trgm ON learners "
        "USING gin (public.immutable_unaccent(first_name) gin_trgm_ops)"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_learners_first_name_unaccent_trgm', table_name='learners')
    op.drop_index('ix_learners_last_name_unaccent_trgm', table_name='learners')
    op.create_index('ix_learners_last_name_trgm', 'learners', ['last_name'], unique=False, postgresql_using='gin', postgresql_ops={'last_name': 'gin_trgm_ops'})
    op.create_index('ix_learners_first_name_trgm', 'learners', ['first_name'], unique=False, postgresql_using='gin', postgresql_ops={'first_name': 'gin_trgm_ops'})
    op.execute("DROP FUNCTION public.immutable_unaccent(text)")
    op.execute("DROP EXTENSION IF EXISTS unaccent")
