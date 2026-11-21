package org.futo.inputmethod.latin.uix.theme

import android.graphics.Canvas
import android.graphics.ColorFilter
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.PixelFormat
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.Shader
import android.graphics.drawable.Drawable

/**
 * Frosted glass style key background.
 *
 * The fill is translucent, so whatever is behind the keyboard (usually the theme background photo)
 * shows through the key. On top of that the key gets a bright top edge, a dark bottom edge and a
 * soft shadow underneath, which makes it read as a raised piece of glass. When the key is pressed
 * the shadow collapses and the fill darkens, so the key appears to sink into the keyboard.
 */
class GlassKeyDrawable(
    private val cornerRadiusPx: Float,
    private val edgePx: Float,
    private val shadowPx: Float,
    private val pressed: Boolean,
    private val fillTop: Int,
    private val fillBottom: Int,
    private val edgeTop: Int,
    private val edgeBottom: Int,
    private val shadowColor: Int,
    private val glossColor: Int
) : Drawable() {

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val rect = RectF()
    private val shadowRect = RectF()

    override fun onBoundsChange(bounds: Rect) {
        rect.setEmpty()
        shadowRect.setEmpty()
    }

    private fun bodyRect(b: Rect): RectF {
        val sideInset = shadowPx * 0.35f
        rect.set(
            b.left + sideInset,
            b.top + edgePx,
            b.right - sideInset,
            b.bottom - if (pressed) sideInset else shadowPx
        )
        return rect
    }

    private fun fillRoundRect(canvas: Canvas, r: RectF, radius: Float, color: Int) {
        paint.shader = null
        paint.style = Paint.Style.FILL
        paint.color = color
        canvas.drawRoundRect(r, radius, radius, paint)
    }

    private fun drawEdge(canvas: Canvas, r: RectF, radius: Float, color: Int, width: Float, top: Boolean) {
        if (width <= 0.0f) return
        canvas.save()
        if (top) {
            canvas.clipRect(r.left, r.top, r.right, r.top + width * 1.5f)
        } else {
            canvas.clipRect(r.left, r.bottom - width * 1.5f, r.right, r.bottom)
        }
        paint.shader = null
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = width
        paint.color = color
        canvas.drawRoundRect(r, radius, radius, paint)
        canvas.restore()
    }

    override fun draw(canvas: Canvas) {
        val b = bounds
        if (b.isEmpty) return

        val r = bodyRect(b)
        if (r.width() <= 0.0f || r.height() <= 0.0f) return

        val radius = cornerRadiusPx.coerceAtMost(minOf(r.width(), r.height()) / 2.0f)

        // Soft shadow, faked with a few offset layers since blur is not available here
        if (!pressed && shadowPx > 0.0f) {
            for (i in 0 until 3) {
                val spread = shadowPx * (i + 1) / 3.0f
                shadowRect.set(r.left, r.top + spread * 0.4f, r.right, r.bottom + spread)
                val alpha = ((shadowColor ushr 24) and 0xFF) * (1.0f - i * 0.28f)
                fillRoundRect(
                    canvas,
                    shadowRect,
                    radius + spread * 0.2f,
                    (shadowColor and 0x00FFFFFF) or (alpha.toInt() shl 24)
                )
            }
        }

        // Glass body
        paint.shader = LinearGradient(
            r.left, r.top, r.left, r.bottom,
            fillTop, fillBottom, Shader.TileMode.CLAMP
        )
        paint.style = Paint.Style.FILL
        paint.color = 0xFFFFFFFF.toInt()
        canvas.drawRoundRect(r, radius, radius, paint)
        paint.shader = null

        // Gloss over the top half
        canvas.save()
        canvas.clipRect(r.left, r.top, r.right, r.top + r.height() * 0.48f)
        fillRoundRect(canvas, r, radius, glossColor)
        canvas.restore()

        drawEdge(canvas, r, radius, edgeTop, edgePx * 2.0f, top = true)
        drawEdge(canvas, r, radius, edgeBottom, edgePx * 1.5f, top = false)
    }

    override fun setAlpha(alpha: Int) {
        paint.alpha = alpha
    }

    override fun setColorFilter(colorFilter: ColorFilter?) {
        paint.colorFilter = colorFilter
    }

    override fun getOpacity(): Int = PixelFormat.TRANSLUCENT
}
