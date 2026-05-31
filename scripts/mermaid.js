'use strict'

/**
 * Convert ```mermaid code blocks to {% mermaid %} tag before markdown rendering.
 * Priority must be <9 because hexo-renderer-markdown-it registers at priority 9.
 * Lower number = runs earlier.
 */
hexo.extend.filter.register('before_post_render', function (data) {
  data.content = data.content.replace(
    /```mermaid *\n?([\s\S]*?)\n```/g,
    (match, code) => '{% mermaid %}\n' + code.trim() + '\n{% endmermaid %}'
  )
  return data
}, 8)
