'use strict'

/**
 * Convert ```mermaid code blocks to {% mermaid %} tag before markdown rendering.
 * This prevents highlight.js from turning them into <figure class="highlight plaintext">
 * which Butterfly's client-side codeToMermaid() can't find.
 */
hexo.extend.filter.register('before_post_render', function (data) {
  data.content = data.content.replace(
    /```mermaid *\n?([\s\S]*?)\n```/g,
    (match, code) => '{% mermaid %}\n' + code.trim() + '\n{% endmermaid %}'
  )
  return data
})
