import reflex as rx

config = rx.Config(
    app_name="frontendv2",
    frontend_port=3001,
    backend_port=8001,
    plugins=[
        rx.plugins.TailwindV4Plugin(),
    ],
)
