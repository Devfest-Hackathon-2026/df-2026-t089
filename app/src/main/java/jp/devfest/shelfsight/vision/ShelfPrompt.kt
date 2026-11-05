package jp.devfest.shelfsight.vision

import jp.devfest.shelfsight.model.ProductCategory

internal object ShelfPrompt {
    val text: String = buildString {
        appendLine("You see a photo taken inside a Japanese convenience store.")
        appendLine("Find the shelf areas that hold products and group them by product type.")
        appendLine("Report at most 8 areas. Merge neighboring products of the same type into one area.")
        appendLine("Use only these category values:")
        appendLine(ProductCategory.entries.joinToString(", ") { it.id })
        appendLine("Respond with only a JSON array and no other text, in this format:")
        appendLine("""[{"category":"coffee","box_2d":[ymin,xmin,ymax,xmax]}]""")
        appendLine("box_2d values are integers from 0 to 1000 relative to the image size.")
        append("If no products are visible, respond with [].")
    }
}
