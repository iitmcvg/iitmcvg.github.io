---
title: "Ongoing Projects"
layout: single
permalink: "/ongoing-projects/"
author_profile: true
---

{% for item in site.ongoing-projects %}
  <h3><a href="{{ item.url }}">{{ item.title }}</a></h3>
  <p>{{ item.description }}</p>
{% endfor %}
