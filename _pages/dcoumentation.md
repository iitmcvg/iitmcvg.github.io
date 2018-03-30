---
title: "Documentation"
layout: single
permalink: "/documentation/"

---
## Computer Vision
{% for item in site.documentation %}
  {% if item.category=="computer_vision" %}
  <h3><a href="{{ item.url }}">{{ item.title }}</a></h3>
  <p>{{ item.description }}</p>
  {% endif %}
{% endfor %}

## Machine Learning

{% for item in site.documentation %}
  {% if item.category=="machine_learning" %}
  <h3><a href="{{ item.url }}">{{ item.title }}</a></h3>
  <p>{{ item.description }}</p>
  {% endif %}
{% endfor %}
