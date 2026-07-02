"""Core shared objects for the Amref Help Desk RAG application.

This package holds singletons and utilities that are needed by *both*
``main.py`` and the route/dependency layer.  Nothing in this package
may import from ``backend.app.main`` – that is the rule that prevents
the circular-import error.

Public surface
--------------
- ``backend.app.core.limiter``  →  the ``slowapi`` ``Limiter`` instance
"""
