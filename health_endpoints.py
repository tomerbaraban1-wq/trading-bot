# Health Check Endpoints for TaskMonitor
# Add these endpoints to your main.py routes (around line 548, after app = FastAPI(...))

from fastapi import FastAPI
from fastapi.responses import JSONResponse  # FIX: JSONResponse lives in fastapi.responses, not fastapi
from task_monitor import get_monitor
import asyncio

async def setup_health_endpoints(app: FastAPI):
    """
    Setup health check endpoints for TaskMonitor

    Call this in your main.py startup:
        app = FastAPI(...)
        await setup_health_endpoints(app)
    """

    @app.get("/monitor/health")
    async def monitor_health():
        """
        Real-time health status of all monitored tasks

        Returns:
        {
            "alive": 48,
            "dead": 0,
            "tasks": {
                "heartbeat_loop": {"alive": true, "restarts": 0, "uptime_seconds": 3600},
                "sentiment_monitor": {"alive": true, "restarts": 0, "uptime_seconds": 3600},
                ...
            }
        }
        """
        monitor = get_monitor()
        if monitor is None:
            return JSONResponse(
                status_code=503,
                content={"error": "Monitor not initialized", "status": "unavailable"}
            )

        try:
            status = await monitor.get_status()
            return JSONResponse(content=status, status_code=200)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e), "status": "error"}
            )

    @app.get("/monitor/alive")
    async def monitor_alive():
        """Simple alive check for heartbeat/monitoring tools"""
        monitor = get_monitor()
        if monitor is None:
            return JSONResponse({"status": "unavailable"}, status_code=503)

        try:
            status = await monitor.get_status()
            total_tasks = status.get("alive", 0) + status.get("dead", 0)

            if status.get("dead", 0) > 0:
                return JSONResponse({
                    "status": "degraded",
                    "alive": status.get("alive"),
                    "dead": status.get("dead"),
                    "total": total_tasks
                }, status_code=200)

            return JSONResponse({
                "status": "healthy",
                "alive": status.get("alive"),
                "total": total_tasks
            }, status_code=200)
        except Exception as e:
            return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

    @app.get("/monitor/tasks")
    async def monitor_tasks():
        """Get detailed task list with individual status"""
        monitor = get_monitor()
        if monitor is None:
            return JSONResponse(
                status_code=503,
                content={"error": "Monitor not initialized"}
            )

        try:
            status = await monitor.get_status()
            tasks = status.get("tasks", {})

            # Sort by alive status
            sorted_tasks = {
                "alive": {},
                "dead": {}
            }

            for task_name, task_info in tasks.items():
                if task_info.get("alive"):
                    sorted_tasks["alive"][task_name] = task_info
                else:
                    sorted_tasks["dead"][task_name] = task_info

            return JSONResponse(content=sorted_tasks, status_code=200)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e)}
            )

    @app.get("/monitor/summary")
    async def monitor_summary():
        """Get a summary for monitoring dashboards"""
        monitor = get_monitor()
        if monitor is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unavailable",
                    "total_tasks": 0,
                    "alive_tasks": 0,
                    "dead_tasks": 0
                }
            )

        try:
            status = await monitor.get_status()
            alive = status.get("alive", 0)
            dead = status.get("dead", 0)
            total = alive + dead

            # Determine health status
            if dead == 0:
                health = "healthy"
            elif dead < 3:
                health = "degraded"
            else:
                health = "critical"

            return JSONResponse({
                "status": health,
                "total_tasks": total,
                "alive_tasks": alive,
                "dead_tasks": dead,
                "health_percentage": (alive / total * 100) if total > 0 else 0,
                "timestamp": asyncio.get_event_loop().time()
            }, status_code=200)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "error": str(e)
                }
            )

    logger.info("Health check endpoints registered: /monitor/health, /monitor/alive, /monitor/tasks, /monitor/summary")


# ════════════════════════════════════════════════════════════════════════════════════
# HOW TO INTEGRATE INTO main.py
# ════════════════════════════════════════════════════════════════════════════════════
#
# 1. Add this import at the top of main.py:
#    from health_endpoints import setup_health_endpoints
#
# 2. In the lifespan() function, after monitor initialization add:
#    await setup_health_endpoints(app)
#
# 3. Or add these endpoints manually to main.py webhook.py routes
#
# ════════════════════════════════════════════════════════════════════════════════════
