# ==============================================================================
# FaceCheck Banking KYC Biometrics - Master Automation Makefile
# ==============================================================================

.PHONY: help setup start stop restart logs migrate test scan backup restore rotate-secrets reset-dev verify-all

SHELL := /bin/bash
.DEFAULT_GOAL := help

help:
	@echo "=========================================================================="
	@echo "          FaceCheck Local Docker Management Automation"
	@echo "=========================================================================="
	@echo "Available commands:"
	@echo "  make setup          - Provision secrets, build, start, migrate, and prepare stack"
	@echo "  make start          - Start containers in detached mode"
	@echo "  make stop           - Gracefully stop containers"
	@echo "  make restart        - Restart running containers"
	@echo "  make logs           - Follow service logs (e.g. make logs service=api)"
	@echo "  make migrate        - Execute pending database migrations"
	@echo "  make test           - Execute all tests inside Docker API container"
	@echo "  make scan           - Run comprehensive security and dependency audit"
	@echo "  make backup         - Create encrypted database backup"
	@echo "  make restore        - Restore database from encrypted backup (file=...)"
	@echo "  make rotate-secrets - Rotate database and API secrets with zero downtime"
	@echo "  make reset-dev      - Permanently wipe dev containers and volumes"
	@echo "  make verify-all     - Run full end-to-end verification and report summary"
	@echo "=========================================================================="

setup:
	@bash scripts/setup.sh

start:
	@bash scripts/start.sh

stop:
	@bash scripts/stop.sh

restart:
	@bash scripts/restart.sh

logs:
	@bash scripts/logs.sh $(service)

migrate:
	@bash scripts/migrate.sh

test:
	@bash scripts/test.sh $(target)

scan:
	@bash scripts/security_scan.sh

backup:
	@bash scripts/backup.sh

restore:
	@bash scripts/restore.sh $(file)

rotate-secrets:
	@bash scripts/rotate_secrets.sh

reset-dev:
	@bash scripts/reset_dev.sh $(flag)

verify-all:
	@bash scripts/verify_all.sh
