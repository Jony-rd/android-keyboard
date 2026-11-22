package org.futo.inputmethod.latin.uix.theme

import android.graphics.Canvas
import android.graphics.ColorFilter
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.PixelFormat
import android.graphics.RadialGradient
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.Shader
import android.graphics.drawable.Drawable

/**
 * Crystal water drop style key background.
 *
 * The body is almost clear so the theme background photo stays visible through the key. What sells
 * the "droplet" is light, not paint: a bright hairline along the top edge with a faint glow under
 * it, a small specular highlight in the upper left corner, a hairline rim so the key still has an
 * edge over a busy photo, and a soft shadow underneath that lifts it off the keyboard. On press the
 * shadow collapses and the body sinks slightly, so the key looks like it is being pushed down.
 */
class CrystalKeyDrawable(
    private val cornerRadiusPx: Float,
    private val shadowPx: Float,
    private val tint: Int,
    private val pressed: Boolean
) : Drawable() {

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val rect = RectF()
    private val shadowRect = RectF()
    private val shineRect = RectF()

    override fun onBoundsChange(bounds: Rect) {
        rect.setEmpty()
        shadowRect.setEmpty()
        shineRect.setEmpty()
    }

    private fun alphaColor(color: Int, alpha: Float): Int {
        val a = ((((color ushr 24) and 0xFF) * alpha).toInt()).coerceIn(0, 255)
        return (color and 0x00FFFFFF) or (a shl 24)
    }

    private fun bodyRect(b: Rect): RectF {
        val sideInset = shadowPx * 0.30f
        val sink = if (pressed) shadowPx * 0.55f else 0.0f
        rect.set(
            b.left + sideInset,
            b.top + shadowPx * 0.22f + sink,
            b.right - sideInset,
            b.bottom - shadowPx * 0.85f + sink * 0.5f
        )
        return rect
    }

    private fun fillRoundRect(canvas: Canvas, r: RectF, radius: Float, color: Int) {
        paint.shader = null
        paint.style = Paint.Style.FILL
        paint.color = color
        canvas.drawRoundRect(r, radius, radius, paint)
    }

    private fun strokeEdge(
        canvas: Canvas,
        r: RectF,
        radius: Float,
        color: Int,
        width: Float,
        top: Boolean
    ) {
        if (width <= 0.0f) return
        canvas.save()
        if (top) {
            canvas.clipRect(r.left, r.top, r.right, r.top + width * 1.4f)
        } else {
            canvas.clipRect(r.left, r.bottom - width * 1.4f, r.right, r.bottom)
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
        val edgePx = (r.height() * 0.02f).coerceIn(1.0f, 2.0f)

        // Soft shadow underneath: a couple of offset layers fakes the blur we cannot do here
        if (!pressed && shadowPx > 0.0f) {
            for (i in 0 until 2) {
                val spread = shadowPx * (i + 1) / 2.0f
                shadowRect.set(
                    r.left + spread * 0.1f,
                    r.top + spread * 0.25f,
                    r.right - spread * 0.1f,
                    r.bottom + spread * 0.9f
                )
                fillRoundRect(
                    canvas,
                    shadowRect,
                    radius,
                    alphaColor(0xFF000000.toInt(), if (i == 0) 0.20f else 0.10f)
                )
            }
        }

        // Body: almost clear white gradient so the background photo shows through
        val bodyAlpha = if (pressed) 1.35f else 1.0f
        paint.shader = LinearGradient(
            r.left, r.top, r.left, r.bottom,
            alphaColor(0xFFFFFFFF.toInt(), (0.13f * bodyAlpha).coerceAtMost(1.0f)),
            alphaColor(0xFFFFFFFF.toInt(), (0.04f * bodyAlpha).coerceAtMost(1.0f)),
            Shader.TileMode.CLAMP
        )
        paint.style = Paint.Style.FILL
        paint.color = 0xFFFFFFFF.toInt()
        canvas.drawRoundRect(r, radius, radius, paint)
        paint.shader = null

        // Faint tint of the theme accent, kept low so the photo still reads through
        fillRoundRect(canvas, r, radius, alphaColor(tint, if (pressed) 0.14f else 0.08f))

        // Specular highlight in the upper left corner, like light caught in a drop of water
        val shineLeft = r.left + r.width() * 0.10f
        val shineTop = r.top + r.height() * 0.10f
        val shineW = r.width() * 0.42f
        val shineH = r.height() * 0.34f
        shineRect.set(shineLeft, shineTop, shineLeft + shineW, shineTop + shineH)
        paint.shader = RadialGradient(
            shineRect.centerX(), shineRect.centerY(), shineH * 0.85f,
            intArrayOf(
                alphaColor(0xFFFFFFFF.toInt(), if (pressed) 0.16f else 0.30f),
                alphaColor(0xFFFFFFFF.toInt(), 0.0f)
            ),
            floatArrayOf(0.0f, 1.0f),
            Shader.TileMode.CLAMP
        )
        paint.style = Paint.Style.FILL
        canvas.drawOval(shineRect, paint)
        paint.shader = null

        // Refraction edges: a bright top hairline with a glow, a darker bottom hairline
        strokeEdge(
            canvas, r, radius,
            alphaColor(0xFFFFFFFF.toInt(), 0.10f),
            edgePx * 3.0f, top = true
        )
        strokeEdge(
            canvas, r, radius,
            alphaColor(0xFFFFFFFF.toInt(), if (pressed) 0.35f else 0.55f),
            edgePx * 1.2f, top = true
        )
        strokeEdge(
            canvas, r, radius,
            alphaColor(0xFF000000.toInt(), if (pressed) 0.18f else 0.26f),
            edgePx, top = false
        )

        // Hairline rim so the key keeps a visible edge over a busy photo
        paint.shader = null
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = edgePx * 0.8f
        paint.color = alphaColor(0xFFFFFFFF.toInt(), if (pressed) 0.22f else 0.14f)
        canvas.drawRoundRect(r, radius, radius, paint)
        paint.color = alphaColor(tint, if (pressed) 0.30f else 0.18f)
        canvas.drawRoundRect(r, radius, radius, paint)
    }

    override fun setAlpha(alpha: Int) {
        paint.alpha = alpha
    }

    override fun setColorFilter(colorFilter: ColorFilter?) {
        paint.colorFilter = colorFilter
    }

    override fun getOpacity(): Int = PixelFormat.TRANSLUCENT
}
