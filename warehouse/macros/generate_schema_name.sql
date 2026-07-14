{# Use the custom schema name verbatim (staging/intermediate/marts) instead of
   dbt's default of prefixing it with the target schema (e.g. main_marts).
   This keeps table references stable: marts.mart_ml_features, etc. #}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- if custom_schema_name is none -%}
        {{ target.schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}
