-- =============================================================================
-- Legal-CoPilot — PostgreSQL Initialization Script
-- Runs once on first container start via docker-entrypoint-initdb.d
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1. Extensions
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";      -- UUID primary keys
CREATE EXTENSION IF NOT EXISTS "vector";          -- pgvector for embeddings
CREATE EXTENSION IF NOT EXISTS "pg_trgm";         -- trigram search for case titles
CREATE EXTENSION IF NOT EXISTS "btree_gin";       -- GIN indexes on composite fields

-- ---------------------------------------------------------------------------
-- 2. Application Role (least-privilege)
--    The app connects as 'vakil_app' — NOT the superuser.
-- ---------------------------------------------------------------------------
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'vakil_app') THEN
    CREATE ROLE vakil_app WITH LOGIN PASSWORD 'vakil_app_secret';
  END IF;
END
$$;

GRANT CONNECT ON DATABASE vakildb TO vakil_app;
GRANT USAGE ON SCHEMA public TO vakil_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO vakil_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO vakil_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO vakil_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT USAGE, SELECT ON SEQUENCES TO vakil_app;

-- ---------------------------------------------------------------------------
-- 3. RLS Helper Function
--    The app sets `app.current_user_id` on each connection/session.
--    All RLS policies use this to isolate rows per lawyer.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION current_app_user_id() RETURNS uuid
  LANGUAGE sql STABLE
  AS $$
    SELECT current_setting('app.current_user_id', true)::uuid;
  $$;

-- ---------------------------------------------------------------------------
-- 4. Audit Timestamp Helper
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$;

-- ---------------------------------------------------------------------------
-- NOTE: Table DDL is managed by Alembic migrations (not here).
--       This script only sets up extensions, roles, and helper functions
--       that must exist before migrations run.
-- ---------------------------------------------------------------------------