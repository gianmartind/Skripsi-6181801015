import React, { useEffect, useMemo, useState } from "react"
import ImageItem from "./image_item"

const ResultList = ({ result_list, changeImageDisplay }) => {
    const [selection, setSelection] = useState(Array(10).fill(false))

    const changeSelection = (i) => {
        const arr = Array(10).fill(false)
        arr[i] = true
        changeImageDisplay(result_list[i])
        setSelection(arr)
    }

    useEffect(() => {
        changeSelection(0)
    }, [result_list])

    const imageItems = useMemo(() => {
        let items = []
        for(let i = 0; i < result_list.length; i++) {
            let data = {
                image_name: result_list[i][0],
                total_weight: result_list[i][1]
            }
            items.push(<ImageItem key={i} data={data} is_selected={selection[i]} clickFunction={() => {changeSelection(i)}}/>)
        }
        return items
    }, [selection, result_list])

    return (
        <div>
            {imageItems}
        </div>
    )
}

export default ResultList