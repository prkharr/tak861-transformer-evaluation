-- Read-only recovery of the procedure signatures supplied in the project evidence.
-- Run with the existing authorized Snowflake role/connection.
-- Return the complete PROCEDURE_DDL text, not a summary or screenshot.
-- These statements do not call, create, replace or train any procedure/model.
-- Keep returned procedure text within the authorized project review workflow.

SELECT GET_DDL('PROCEDURE',
  'DSVC_TAKEDA_TA_PRIVATE.DS_ML_DEV.FEATURE_CREATION_PROC(VARCHAR,VARCHAR,VARCHAR,VARCHAR,VARCHAR,VARCHAR,VARCHAR)')
  AS PROCEDURE_DDL;

SELECT GET_DDL('PROCEDURE',
  'DSVC_TAKEDA_TA_PRIVATE.DS_ML_DEV.FEATURE_ENGINEERING_PROD(VARCHAR,VARCHAR,VARCHAR)')
  AS PROCEDURE_DDL;

SELECT GET_DDL('PROCEDURE',
  'DSVC_TAKEDA_TA_PRIVATE.DS_ML_DEV.FEATURE_CREATION_PROD(VARCHAR,VARCHAR,VARCHAR,VARCHAR)')
  AS PROCEDURE_DDL;
