# -*- coding: utf-8 -*-
"""
Custom Leaflet ScaleControl plugin for Folium.
"""
from branca.element import MacroElement
from jinja2 import Template


class ScaleControl(MacroElement):
    """Custom Folium Scale Control integrating Leaflet's L.Control.Scale."""

    def __init__(
        self,
        position: str = "bottomleft",
        metric: bool = True,
        imperial: bool = False,
        max_width: int = 100,
    ):
        super().__init__()
        self._template = Template(u"""
            {% macro script(this, kwargs) %}
                L.control.scale({
                    position: '{{ this.position }}',
                    metric: {{ this.metric | lower }},
                    imperial: {{ this.imperial | lower }},
                    maxWidth: {{ this.max_width }}
                }).addTo({{ this._parent.get_name() }});
            {% endmacro %}
        """)
        self.position = position
        self.metric = metric
        self.imperial = imperial
        self.max_width = max_width
