.PHONY: build up down shell clean logs help

# Default target
help:
	@echo "Available commands:"
	@echo "  make build        - Build Docker image"
	@echo "  make up          - Start container in background"
	@echo "  make down        - Stop and remove container"
	@echo "  make shell       - Enter container shell"
	@echo "  make logs        - View container logs"
	@echo "  make clean       - Stop container and clean up"
	@echo "  make rebuild     - Rebuild image from scratch"

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

shell:
	docker-compose exec yolo bash

logs:
	docker-compose logs -f yolo

clean:
	docker-compose down -v
	docker system prune -f

rebuild: clean build

